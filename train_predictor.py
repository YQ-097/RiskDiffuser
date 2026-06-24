import os
import io
import torch
import argparse
from torch import optim
from timm.utils import ModelEma
from mmengine import fileio
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from riskdiffuser.model.riskdiffuser import RiskDiffuser
from riskdiffuser.model.risknet import SharedEncoderRiskNet

from riskdiffuser.utils.train_utils import resume_model_risk, set_seed, save_model, resume_model
from riskdiffuser.utils.normalizer import ObservationNormalizer, StateNormalizer
from riskdiffuser.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from riskdiffuser.utils.tb_log import TensorBoardLogger as Logger
from riskdiffuser.utils.data_augmentation import StatePerturbation
from riskdiffuser.utils.dataset import RiskDiffuserData
from riskdiffuser.utils import ddp

from riskdiffuser.train_epoch import train_epoch

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "riskdiffuser-training")', default="riskdiffuser-training")
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default=".")

    # Data
    parser.add_argument('--train_set', type=str, help='path to train data', default=None)
    parser.add_argument('--train_set_list', type=str, help='data list of train data', default=None)

    parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    
    # DataLoader parameters
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    parser.add_argument('--normalization_file_path', default='normalization.json', help='filepath of normalizaiton.json', type=str)
    parser.add_argument('--use_data_augment', default=True, type=boolean)
    parser.add_argument('--num_workers', default=4, type=int)#4
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    
    # Training
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=10)#30
    parser.add_argument('--save_utd', type=int, help='save frequency', default=2)#20
    parser.add_argument('--batch_size', type=int, help='batch size (default: 2048)', default=256) #2048  #64不冻结的情况下5090单卡只能64
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=5)
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    parser.add_argument('--alpha_planning_loss', type=float, help='coefficient of planning loss (default: 1.0)', default=1.0)
    parser.add_argument('--use_jerk_loss', default=True, type=boolean, help='use ego jerk smoothness loss on predicted future trajectory')
    parser.add_argument('--alpha_jerk_loss', type=float, default=0.0002, help='weight of ego jerk smoothness loss') #0.0002
    parser.add_argument('--use_risk_order_loss', default=True, type=boolean, help='use reference-vs-random risk order consistency loss')
    parser.add_argument('--alpha_risk_order_loss', type=float, default=0.1, help='weight of risk order consistency loss')#0.1
    parser.add_argument('--risk_order_rho_threshold', type=float, default=0.05, help='ignore risk pairs with too small rho gap')
    parser.add_argument('--random_rho_min', type=float, default=0.0, help='minimum random risk level')
    parser.add_argument('--random_rho_max', type=float, default=1.0, help='maximum random risk level')

    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')

    parser.add_argument('--use_ema', default=True, type=boolean)

    # Model
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    parser.add_argument('--num_heads', type=int, help='number of multi-head', default=6)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=192)
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model [x_start, score]', choices=['score', 'x_start'], default='x_start')

    # decoder
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--resume_model_path', type=str, help='path to resume model', default=None)
    parser.add_argument('--risk_model_path', type=str, help='path to pretrained risk net checkpoint dir or .pth', default=None)
    parser.add_argument('--use_risk_model_infer', default=True, type=boolean, help='use frozen risk net to infer risk_level for risk adapter training')
    parser.add_argument('--shuffle_inferred_rho_within_batch', default=False, type=boolean, help='shuffle inferred RiskNet rho across samples within each batch before feeding risk_level')

    parser.add_argument('--use_wandb', default=False, type=boolean)
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    parser.add_argument('--ddp', default=False, type=boolean, help='use ddp or not')
    parser.add_argument('--port', default='22323', type=str, help='port')

    #parser.add_argument('--rho', default='0', type=float, help='risk level value') #risk level value

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    return args


def _resolve_ckpt_path(path, default_name='latest.pth'):
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, default_name),
            os.path.join(path, 'model.pth'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No checkpoint found under directory: {path}")
    return path


def load_frozen_risk_model(args, device):
    if args.risk_model_path is None:
        return None

    base_model = RiskDiffuser(args).to(device)
    risk_model = SharedEncoderRiskNet(base_model.encoder, args).to(device)

    ckpt_path = _resolve_ckpt_path(args.risk_model_path)
    ckpt_bytes = fileio.get(ckpt_path)
    with io.BytesIO(ckpt_bytes) as f:
        ckpt = torch.load(f, map_location=device, weights_only=False)

    print(f"Loading RiskNet checkpoint from {ckpt_path}")

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    msg = risk_model.load_state_dict(state_dict, strict=False)

    print("RiskNet load done")
    print("RiskNet missing keys:", msg.missing_keys)
    print("RiskNet unexpected keys:", msg.unexpected_keys)

    if args.use_ema and 'ema_state_dict' in ckpt:
        ema = ModelEma(risk_model, decay=0.999, device=device)
        ema_state = ckpt['ema_state_dict']
        if isinstance(ema_state, dict) and 'module' in ema_state:
            ema_state = ema_state['module']
        ema_state = {k.replace('module.', ''): v for k, v in ema_state.items()}
        try:
            ema.ema.load_state_dict(ema_state, strict=False)
            risk_model = ema.ema
            print("RiskNet EMA load done")
        except Exception as e:
            print(f"RiskNet EMA load failed, fallback to raw model. Reason: {e}")

    risk_model.eval()
    for param in risk_model.parameters():
        param.requires_grad = False

    return risk_model


def resolve_runtime_device(args, rank):
    if args.device == 'cuda':
        return torch.device(f'cuda:{rank}')
    return torch.device(args.device)

def model_training(args):

    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_model_path is not None:
            save_path = args.resume_model_path
        else:
            from datetime import datetime
            time = datetime.now()
            time = time.strftime("%Y-%m-%d-%H:%M:%S")

            save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
            os.makedirs(save_path, exist_ok=True)

        # Save args
        args_dict = vars(args)
        args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

        from mmengine.fileio import dump
        dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    runtime_device = resolve_runtime_device(args, rank)
    
    # set up data loaders
    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    train_set= RiskDiffuserData(args.train_set, args.train_set_list, args.agent_num, args.predicted_neighbor_num, args.future_len)
    train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size//ddp.get_world_size(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
   
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    riskdiffuser = RiskDiffuser(args)
    riskdiffuser = riskdiffuser.to(runtime_device)

    risk_model = None
    if args.use_risk_model_infer:
        risk_model = load_frozen_risk_model(args, runtime_device)
        if global_rank == 0:
            if risk_model is None:
                print("RiskNet infer disabled: fallback to dataset risk_level")
            else:
                print("RiskNet infer enabled: using frozen RiskNet outputs as risk_level")

    if args.ddp:
        riskdiffuser = DDP(riskdiffuser, device_ids=[rank])

    model_ema = None
    if args.use_ema:
        model_ema = ModelEma(
            riskdiffuser,
            decay=0.999,
            device=args.device,
        )
    
    if global_rank == 0:
        print("Model Params: {}".format(sum(p.numel() for p in ddp.get_model(riskdiffuser, args.ddp).parameters())))

    # optimizer
    params = [{'params': ddp.get_model(riskdiffuser, args.ddp).parameters(), 'lr': args.learning_rate}]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        #riskdiffuser, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path, riskdiffuser, optimizer, scheduler, model_ema, args.device)
        riskdiffuser, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model_risk(args.resume_model_path, riskdiffuser,train_epochs, args.device)
    else:
        init_epoch = 0
        wandb_id = None

    # logger
    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank) 

    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_total_loss = train_epoch(
            train_loader,
            riskdiffuser,
            optimizer,
            args,
            model_ema,
            aug,
            risk_model=risk_model,
        )
        


        if global_rank == 0:
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch+1)
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch+1)

            if (epoch+1) % args.save_utd == 0:
                # save model at the end of epoch
                ema_to_save = model_ema.ema if model_ema is not None else ddp.get_model(riskdiffuser, args.ddp)
                save_model(riskdiffuser, optimizer, scheduler, save_path, epoch, train_total_loss, wandb_logger.id, ema_to_save)
                print(f"Model saved in {save_path}\n")

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)

if __name__ == "__main__":

    args = get_args()
    
    # Run
    model_training(args)
