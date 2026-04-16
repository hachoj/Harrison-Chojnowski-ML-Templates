# Harrison Chojnowski ML Templates

Personal collection of reusable ML training templates.

## Templates

### `ddp-training-template/`
Barebones PyTorch DDP training script with:
- Multi-GPU setup via `torchrun` / `torch.distributed`
- Muon + AdamW optimizer with param group routing (2D weights → Muon, biases/norms → AdamW)
- OmegaConf config loading from `configs/config.yaml`
- WandB logging (rank 0 only)

**Launch:**
```bash
torchrun --nproc_per_node=<num_gpus> train.py
```
