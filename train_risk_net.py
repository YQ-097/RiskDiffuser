import os
import io
import argparse
from datetime import datetime

import torch
from torch import device, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEma
from mmengine import fileio
from mmengine.fileio import dump

from riskdiffuser.model.riskdiffuser import RiskDiffuser
from riskdiffuser.utils.dataset import RiskDiffuserData
from riskdiffuser.utils.normalizer import ObservationNormalizer, StateNormalizer
from riskdiffuser.utils.data_augmentation import StatePerturbation
from riskdiffuser.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from riskdiffuser.utils.tb_log import TensorBoardLogger as Logger
from riskdiffuser.utils.train_utils import set_seed, get_epoch_mean_loss

from riskdiffuser.model.risknet import (
    SharedEncoderRiskNet,
    RiskHeadLoss,
    build_future_risk_teacher,
    TeacherRiskCalibratorTorch,
)


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Train Risk Assessment Network")

    parser.add_argument("--name", type=str, default="risk-net-training")
    parser.add_argument("--save_dir", type=str, default=".")

    # data
    parser.add_argument("--train_set", type=str, default=None)
    parser.add_argument("--train_set_list", type=str, default=None)

    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--time_len", type=int, default=21)

    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=32)

    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_state_dim", type=int, default=12)
    parser.add_argument("--lane_num", type=int, default=70)

    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_state_dim", type=int, default=12)
    parser.add_argument("--route_num", type=int, default=25)

    parser.add_argument("--predicted_neighbor_num", type=int, default=10)

    # dataloader
    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--normalization_file_path", type=str, default="normalization.json")
    parser.add_argument("--use_data_augment", type=boolean, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--save_utd", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", type=boolean, default=True)

    # encoder config
    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument(
        "--diffusion_model_type",
        type=str,
        choices=["score", "x_start"],
        default="x_start",
    )
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)

    # loading
    parser.add_argument(
        "--pretrained_riskdiffuser_ckpt",
        type=str,
        default=None,
        help="path to pretrained riskdiffuser checkpoint dir or .pth",
    )
    parser.add_argument(
        "--resume_risk_ckpt",
        type=str,
        default=None,
        help="resume risk net training from this checkpoint dir or .pth",
    )

    # freeze
    parser.add_argument("--freeze_encoder", type=boolean, default=True)

    # logging
    parser.add_argument("--use_wandb", type=boolean, default=False)
    parser.add_argument("--notes", type=str, default="")

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args


def build_inputs_from_batch(batch, device):
    inputs = {
        "ego_current_state": batch[0].to(device),
        "neighbor_agents_past": batch[2].to(device),

        "lanes": batch[4].to(device),
        "lanes_speed_limit": batch[5].to(device),
        "lanes_has_speed_limit": batch[6].to(device),

        "route_lanes": batch[7].to(device),
        "route_lanes_speed_limit": batch[8].to(device),
        "route_lanes_has_speed_limit": batch[9].to(device),

        "static_objects": batch[10].to(device),
    }

    ego_future_gt = batch[1].to(device)
    neighbors_future_gt = batch[3].to(device)

    return inputs, ego_future_gt, neighbors_future_gt


def save_risk_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema=None):
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": scheduler.state_dict(),
        "loss": train_loss,
        "wandb_id": wandb_id,
    }

    if ema is not None:
        save_dict["ema_state_dict"] = ema.state_dict()

    with io.BytesIO() as f:
        torch.save(save_dict, f)
        fileio.put(f.getvalue(), f"{save_path}/risk_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth")
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")


def _resolve_ckpt_path(path):
    if os.path.isdir(path):
        candidates = [
            #os.path.join(path, "latest.pth"),
            os.path.join(path, "model.pth"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No checkpoint found under directory: {path}")
    return path


def _load_ckpt_bytes(path, map_location="cpu"):
    real_path = _resolve_ckpt_path(path)
    ckpt_bytes = fileio.get(real_path)
    with io.BytesIO(ckpt_bytes) as f:
        ckpt = torch.load(f, map_location=map_location, weights_only=False)
    return ckpt, real_path


def load_riskdiffuser_encoder_to_risknet(risk_model, base_model, ckpt_path, device="cpu"):
    """
    新开 RiskNet 训练时使用：
    从 RiskDiffuser checkpoint 加载权重到 base_model，
    然后把 encoder 权重复制到 risk_model.encoder。
    不恢复 optimizer / scheduler / epoch。
    """
    ckpt, real_path = _load_ckpt_bytes(ckpt_path, map_location=device)
    print(f"Loading RiskDiffuser checkpoint from: {real_path}")

    state_dict = ckpt.get("model", ckpt)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    msg = base_model.load_state_dict(clean_state_dict, strict=False)
    print("RiskDiffuser model load done")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    risk_model.encoder.load_state_dict(base_model.encoder.state_dict(), strict=True)
    print("Encoder weights copied to RiskNet encoder")

    return risk_model


def resume_risk_net_training(path, model, optimizer=None, scheduler=None, device="cpu", use_ema=True):
    """
    RiskNet 自己的续训：
    恢复 model / optimizer / scheduler / ema / epoch。
    """
    ckpt, real_path = _load_ckpt_bytes(path, map_location=device)
    print(f"Resuming RiskNet checkpoint from: {real_path}")

    state_dict = ckpt.get("model", ckpt)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(clean_state_dict, strict=False)
    print("RiskNet model load done")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Optimizer load done")
        except Exception as e:
            print(f"Optimizer load failed, skipped. Reason: {e}")

    if scheduler is not None and "schedule" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["schedule"])
            print("Scheduler load done")
        except Exception as e:
            print(f"Scheduler load failed, skipped. Reason: {e}")

    ema = None
    if use_ema:
        ema = ModelEma(model, decay=0.999, device=device)
        if "ema_state_dict" in ckpt:
            try:
                ema_state = ckpt["ema_state_dict"]
                clean_ema_state = {k.replace("module.", ""): v for k, v in ema_state.items()}
                ema.ema.load_state_dict(clean_ema_state, strict=False)
                ema.ema.eval()
                for p in ema.ema.parameters():
                    p.requires_grad_(False)
                print("EMA load done")
            except Exception as e:
                print(f"EMA load failed, using fresh EMA. Reason: {e}")
        else:
            print("No EMA found in checkpoint, using fresh EMA.")

    init_epoch = ckpt.get("epoch", 0)
    wandb_id = ckpt.get("wandb_id", None)

    return model, optimizer, scheduler, init_epoch, wandb_id, ema


def freeze_encoder_if_needed(model, freeze_encoder=False):
    if not freeze_encoder:
        return

    for param in model.encoder.parameters():
        param.requires_grad = False

    print("Encoder frozen. Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def train_epoch_risk(data_loader, model, optimizer, criterion, args, ema=None, aug=None):
    epoch_loss = []
    model.train()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    calibrator = TeacherRiskCalibratorTorch(device=device)

    with tqdm(data_loader, desc="Training RiskNet", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future_gt, neighbors_future_gt = build_inputs_from_batch(batch, device)

            if aug is not None:
                inputs, ego_future_gt, neighbors_future_gt = aug(inputs, ego_future_gt, neighbors_future_gt)

            inputs = args.observation_normalizer(inputs)

            
            teacher = build_future_risk_teacher(
                ego_current_state=batch[0].to(device),
                ego_future_gt=ego_future_gt,
                neighbor_agents_past=batch[2].to(device),
                neighbors_future_gt=neighbors_future_gt,
                route_lanes=batch[7].to(device),
                route_lanes_speed_limit=batch[8].to(device),
                route_lanes_has_speed_limit=batch[9].to(device),
                static_objects=batch[10].to(device),
            )
            teacher["risk_target"] = calibrator.transform(teacher["risk_target"])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss_dict = criterion(outputs, teacher)

            total_loss = loss_dict["loss"]
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            data_epoch.set_postfix(loss=f"{total_loss.item():.4f}")
            epoch_loss.append(loss_dict)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)
    print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
    return epoch_mean_loss, epoch_mean_loss["loss"]


def model_training(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"------------- {args.name} -------------")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use device: {device}")

    # 新实验默认新建目录；只有明确 resume 风险网络时才沿用旧目录
    if args.resume_risk_ckpt is not None and os.path.isdir(args.resume_risk_ckpt):
        save_path = args.resume_risk_ckpt
    else:
        time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        save_path = f"{args.save_dir}/training_log/{args.name}/{time_str}/"
        os.makedirs(save_path, exist_ok=True)

    args_dict = vars(args)
    args_dict = {
        k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
        for k, v in args_dict.items()
    }
    dump(args_dict, os.path.join(save_path, "args.json"), file_format="json", indent=4)

    set_seed(args.seed)

    aug = StatePerturbation(augment_prob=args.augment_prob, device=device) if args.use_data_augment else None

    train_set = RiskDiffuserData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    print(f"Dataset Prepared: {len(train_set)} train data\n")

    # 1) 先构建 RiskDiffuser，只是为了得到 encoder 结构
    base_model = RiskDiffuser(args).to(device)

    # 2) 用共享 encoder 构建 RiskNet
    risk_model = SharedEncoderRiskNet(base_model.encoder, args).to(device)

    # 3) 新开 RiskNet 训练：从 RiskDiffuser checkpoint 初始化 encoder
    if args.pretrained_riskdiffuser_ckpt is not None:
        risk_model = load_riskdiffuser_encoder_to_risknet(
            risk_model,
            base_model,
            args.pretrained_riskdiffuser_ckpt,
            device=device,
        )
    else:
        print("Warning: pretrained_riskdiffuser_ckpt is None, RiskNet encoder will use random init.")

    # 4) 按需冻结 encoder
    freeze_encoder_if_needed(risk_model, args.freeze_encoder)

    print("RiskNet Params:", sum(p.numel() for p in risk_model.parameters()))
    print("Trainable Params:", sum(p.numel() for p in risk_model.parameters() if p.requires_grad))

    criterion = RiskHeadLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, risk_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        args.train_epochs,
        args.warm_up_epoch,
    )

    # 新开训练默认从 0 开始
    init_epoch = 0
    wandb_id = None

    model_ema = None
    if args.use_ema:
        model_ema = ModelEma(
            risk_model,
            decay=0.999,
            device=device,
        )

    # 只有你明确提供 risk checkpoint，才做 RiskNet 自己的续训
    if args.resume_risk_ckpt is not None:
        risk_model, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_risk_net_training(
            args.resume_risk_ckpt,
            risk_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_ema=args.use_ema,
        )

    wandb_logger = Logger(
        args.name,
        args.notes,
        args,
        wandb_resume_id=wandb_id,
        save_path=save_path,
        rank=0,
    )
    
    for epoch in range(init_epoch, args.train_epochs):
        print(f"Epoch {epoch + 1}/{args.train_epochs}")

        train_loss, train_total_loss = train_epoch_risk(
            train_loader,
            risk_model,
            optimizer,
            criterion,
            args,
            ema=model_ema,
            aug=aug,
        )

        lr_dict = {"lr": optimizer.param_groups[0]["lr"]}
        wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch + 1)
        wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch + 1)

        if (epoch + 1) % args.save_utd == 0:
            ema_model = model_ema.ema if model_ema is not None else None
            save_risk_model(
                risk_model,
                optimizer,
                scheduler,
                save_path,
                epoch,
                train_total_loss,
                wandb_logger.id,
                ema=ema_model,
            )
            print(f"Model saved in {save_path}\n")

        scheduler.step()


if __name__ == "__main__":
    args = get_args()
    model_training(args)
