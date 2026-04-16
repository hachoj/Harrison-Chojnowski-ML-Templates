import os

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Muon


def build_param_groups(model, lr_adamw, lr_muon, wd):
    muon_params = []
    adamw_decay = []
    adamw_no_decay = []

    no_decay_keywords = {"bias", "norm", "ln_", "layernorm", "embedding"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip_decay = any(kw in name.lower() for kw in no_decay_keywords)
        if p.ndim == 2 and not skip_decay:
            muon_params.append(p)
        elif skip_decay:
            adamw_no_decay.append(p)
        else:
            adamw_decay.append(p)

    return {
        "muon": [{"params": muon_params, "lr": lr_muon}],
        "adamw": [
            {"params": adamw_decay, "lr": lr_adamw, "weight_decay": wd},
            {"params": adamw_no_decay, "lr": lr_adamw, "weight_decay": 0.0},
        ],
    }


def train(model, adamw, muon, cfg, device, rank, world_size, is_main):
    # if cfg.wandb.enabled and is_main:
    #     wandb.init(
    #         project=cfg.wandb.project,
    #         name=cfg.wandb.name,
    #         dir=cfg.wandb.dir,
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #     )
    pass


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    is_main: bool = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        model = torch.hub.load(
            cfg.model.repo_dir,
            cfg.model.base_name,
            source="local",
            weights=cfg.model.weights_path,
        )
    except Exception as e:
        print(f"Error loading local DINOv3 model: {e}")
        print("To fix, ensure the path to dinov3 repo and weights is exact.")
        raise

    model = model.to(device=device)
    model = DDP(model, device_ids=[local_rank])

    # just given how idno is normally loaded, I assume for some reason it
    # default to not being loaded in train mode so I'll just do that once here
    # also set parameters grad = true
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    groups = build_param_groups(
        model,
        lr_adamw=cfg.train.lr_adamw,
        lr_muon=cfg.train.lr_muon,
        wd=cfg.train.weight_decay,
    )
    muon = Muon(groups["muon"])
    adamw = AdamW(groups["adamw"])

    train(
        model=model,
        adamw=adamw,
        muon=muon,
        cfg=cfg,
        device=device,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
    )
