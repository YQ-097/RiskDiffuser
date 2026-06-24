import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json
from torch import optim
from timm.utils import ModelEma
from torch.optim.lr_scheduler import CosineAnnealingLR

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)

    return npz_data

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
    """
    save the model to path
    """
    save_model = {'epoch': epoch + 1, 
                  'model': model.state_dict(), 
                  'ema_state_dict': ema.state_dict(),
                  'optimizer': optimizer.state_dict(), 
                  'schedule': scheduler.state_dict(), 
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    with io.BytesIO() as f:
        torch.save(save_model, f)
        fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")

# def resume_model(path: str, model, optimizer, scheduler, ema, device):
#     """
#     load ckpt from path
#     """
#     path = os.path.join(path, 'latest.pth')
#     ckpt = fileio.get(path)
#     with io.BytesIO(ckpt) as f:
#         ckpt = torch.load(f)

#     # load model
#     try:
#         model.load_state_dict(ckpt['model'])
#     except:
#         model.load_state_dict(ckpt)                   
#     print("Model load done")
    
#     # load optimizer
#     try:
#         optimizer.load_state_dict(ckpt['optimizer'])
#         print("Optimizer load done")
#     except:
#         print("no pretrained optimizer found")
            
#     # load schedule
#     try:
#         scheduler.load_state_dict(ckpt['schedule'])
#         print("Schedule load done")
#     except:
#         print("no schedule found,")
    
#     # load step
#     try:
#         init_epoch = ckpt['epoch']
#         print("Step load done")
#     except:
#         init_epoch = 0

#     # Load wandb id
#     try:
#         wandb_id = ckpt['wandb_id']
#         print("wandb id load done")
#     except:
#         wandb_id = None

#     try:
#         ema.ema.load_state_dict(ckpt['ema_state_dict'])
#         ema.ema.eval()
#         for p in ema.ema.parameters():
#             p.requires_grad_(False)

#         print("ema load done")
#     except:
#         print('no ema shadow found')

#     return model, optimizer, scheduler, init_epoch, wandb_id, ema


def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    Robust checkpoint loader
    Compatible with:
        - DDP / non-DDP
        - model / ema_state_dict
        - added new modules (e.g. risk_adapter)
    """

    path = os.path.join(path, 'latest.pth')
    ckpt_bytes = fileio.get(path)

    with io.BytesIO(ckpt_bytes) as f:
        ckpt = torch.load(f, map_location=device)

    print(f"Loading checkpoint from {path}")

    # ------------------------
    # 1️⃣ Load model weights
    # ------------------------
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # remove DDP "module." prefix
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    msg = model.load_state_dict(clean_state_dict, strict=False)

    print("Model load done")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    # ------------------------
    # 2️⃣ Load optimizer
    # ------------------------
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Optimizer load done")
        except Exception as e:
            print("Optimizer load failed:", e)
    else:
        print("No optimizer found")

    # ------------------------
    # 3️⃣ Load scheduler
    # ------------------------
    if "schedule" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["schedule"])
            print("Schedule load done")
        except Exception as e:
            print("Schedule load failed:", e)
    else:
        print("No scheduler found")

    # ------------------------
    # 4️⃣ Load epoch
    # ------------------------
    init_epoch = ckpt.get("epoch", 0)
    print(f"Start epoch: {init_epoch}")

    # ------------------------
    # 5️⃣ Load wandb id
    # ------------------------
    wandb_id = ckpt.get("wandb_id", None)
    if wandb_id is not None:
        print("wandb id load done")

    # ------------------------
    # 6️⃣ Load EMA
    # ------------------------
    if "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]

        clean_ema_state = {}
        for k, v in ema_state.items():
            if k.startswith("module."):
                clean_ema_state[k[7:]] = v
            else:
                clean_ema_state[k] = v

        try:
            ema.ema.load_state_dict(clean_ema_state, strict=False)
            ema.ema.eval()
            for p in ema.ema.parameters():
                p.requires_grad_(False)
            print("EMA load done")
        except Exception as e:
            print("EMA load failed:", e)
    else:
        print("No EMA found")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema


def resume_model_risk(path: str, model, train_epochs, device):

    path = os.path.join(path, 'model.pth')
    ckpt_bytes = fileio.get(path)

    with io.BytesIO(ckpt_bytes) as f:
        ckpt = torch.load(f, map_location=device,weights_only=False)#,weights_only=False

    print(f"Loading checkpoint from {path}")

    # for name, _ in model.named_parameters():
    #     if "blocks" in name:
    #         print(name)
    # ------------------------
    # 1️⃣ Load model weights
    # Prefer EMA weights for risk fine-tuning if they exist.
    # ------------------------
    # if "ema_state_dict" in ckpt:
    #     state_dict = ckpt["ema_state_dict"]
    #     print("Loading EMA weights into model")
    # else:
    #     state_dict = ckpt.get("model", ckpt)
    #     print("EMA weights not found, falling back to model weights")
    state_dict = ckpt.get("model", ckpt)

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_state_dict[k.replace("module.", "")] = v

    msg = model.load_state_dict(clean_state_dict, strict=False)

    print("Model load done")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    # ------------------------
    # 2️⃣ Freeze backbone
    # ------------------------
    # for p in model.backbone.parameters():
    #     p.requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 2️⃣ 只打开你想训练的模块
    for name, param in model.named_parameters():
        if "risk_adapter" in name:
            param.requires_grad = True
        # if "decoder.decoder.dit.risk_adapter" in name:
        #     param.requires_grad = True
        # elif "decoder.decoder.dit.blocks.2" in name:
        #     param.requires_grad = True
        # elif "decoder.decoder.dit.final_layer" in name:
        #     param.requires_grad = True

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print("Trainable params:")
    for n in trainable_params:
        print(n)

    print("Total trainable:", len(trainable_params))

    # ------------------------
    # 3️⃣ Rebuild optimizer (DON'T load old one)
    # ------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4
    )

    # ------------------------
    # 4️⃣ New scheduler (DON'T load old one)
    # ------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_epochs,
        eta_min=1e-6
    )

    # ------------------------
    # 5️⃣ Fresh EMA
    # ------------------------
    ema = ModelEma(
        model,
        decay=0.999,
        device=device,
    )
    # 7. try loading old ema
    try:
        ema_state = ckpt["ema_state_dict"]
        clean_ema_state = {}
        for k, v in ema_state.items():
            clean_ema_state[k.replace("module.", "")] = v

        ema.ema.load_state_dict(clean_ema_state, strict=False)
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("EMA load done")
        #compare_model_and_ema(model, ema)
    except Exception as e:
        print(f"No compatible EMA found, using fresh EMA. Reason: {e}")

    init_epoch = 0   # 微调重新计数
    wandb_id = None  # 新实验

    return model, optimizer, scheduler, init_epoch, wandb_id, ema



