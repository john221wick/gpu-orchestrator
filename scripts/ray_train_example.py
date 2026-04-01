#!/usr/bin/env python3
"""
ray_train_example.py — Ray Train distributed training demo

Uses TorchTrainer + ScalingConfig. The scheduler sets CUDA_VISIBLE_DEVICES
before launching this script; Ray Train reads that and isolates each worker
to its assigned GPU. Ray injects RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR,
MASTER_PORT into each worker automatically.

Usage (via scheduler):
  gpu-submit scripts/job_ray_train.json

Usage (direct, local):
  python scripts/ray_train_example.py --num-workers 2

Requires: pip install "ray[train]" torch
"""
import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import ray
    import ray.train
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig
except ImportError:
    raise SystemExit(
        "Ray not installed. Run: pip install 'ray[train]'"
    )


# ─── Synthetic dataset ────────────────────────────────────────────────────────
class SyntheticDataset(Dataset):
    def __init__(self, vocab_size=256, seq_len=64, n=2000):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n          = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x, x.clone()  # (input, target)


# ─── Small model ─────────────────────────────────────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, d=128, n_layers=2):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, d)
        self.lstm = nn.LSTM(d, d, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(h)


# ─── Training function (runs inside each Ray worker) ─────────────────────────
def train_func(config):
    """
    This function runs once per worker. Ray Train sets:
      - RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
      - CUDA_VISIBLE_DEVICES (inherited from the driver, restricted by scheduler)
    """
    steps      = config.get("steps", 50)
    batch_size = config.get("batch_size", 16)
    lr         = config.get("lr", 1e-3)
    vocab_size = 256

    ctx        = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    local_rank = ctx.get_local_rank()
    is_main    = (world_rank == 0)

    if is_main:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        print(f"[RAY_TRAIN] world_size={world_size}  CUDA_VISIBLE_DEVICES={gpus}")
        print(f"[RAY_TRAIN] steps={steps}  batch_per_worker={batch_size}")

    # ── Model — Ray Train wraps it in DDP automatically ──
    model = TinyLM(vocab_size=vocab_size)
    model = ray.train.torch.prepare_model(model)

    # ── Dataset + DistributedSampler ──
    dataset = SyntheticDataset(vocab_size=vocab_size, n=steps * batch_size * world_size * 2)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    loader  = ray.train.torch.prepare_data_loader(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    step       = 0
    total_loss = 0.0

    for x, y in loader:
        if step >= steps:
            break

        optimizer.zero_grad()
        logits = model(x)                          # (B, T, vocab)
        loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step       += 1

        if is_main and (step % 10 == 0 or step == 1):
            avg  = total_loss / step
            ppl  = math.exp(avg) if avg < 20 else float('inf')
            print(f"[RAY_TRAIN] Step {step:3d}/{steps}  loss={avg:.4f}  ppl={ppl:.1f}")

        # Report metrics back to Ray (used for fault tolerance / checkpointing)
        ray.train.report({"loss": total_loss / step, "step": step})

    if is_main:
        print(f"[RAY_TRAIN] Training complete  avg_loss={total_loss/max(step,1):.4f}")


# ─── Driver ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers",    type=int,   default=2,    dest="num_workers",
                        help="Number of distributed workers (= num GPUs)")
    parser.add_argument("--steps",          type=int,   default=50)
    parser.add_argument("--batch",          type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--storage-path",   type=str,   default="/tmp/ray-results",
                        dest="storage_path",
                        help="Where Ray stores checkpoints/results")
    args = parser.parse_args()

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[RAY_TRAIN] Starting  num_workers={args.num_workers}  "
          f"CUDA_VISIBLE_DEVICES={gpus}")

    # Ray auto-init with the GPUs visible to this process.
    # The scheduler has already set CUDA_VISIBLE_DEVICES, so Ray only sees
    # the allocated GPUs.
    if not ray.is_initialized():
        ray.init(num_gpus=args.num_workers, ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "steps":      args.steps,
            "batch_size": args.batch,
            "lr":         args.lr,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=torch.cuda.is_available(),
            # resources_per_worker={"GPU": 1} is the default when use_gpu=True
        ),
        run_config=RunConfig(
            name="gpu-sched-ray-train",
            storage_path=args.storage_path,
        ),
    )

    result = trainer.fit()
    print(f"[RAY_TRAIN] Result: {result.metrics}")


if __name__ == "__main__":
    main()
