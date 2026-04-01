#!/usr/bin/env python3
"""
trainer_example.py — HuggingFace Trainer distributed fine-tuning demo

Uses the Trainer API, which auto-detects the distributed environment set up
by torchrun (LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
No manual dist.init_process_group() needed — Trainer handles everything.

Usage (via scheduler):
  gpu-submit scripts/job_hf_trainer.json

Usage (direct):
  torchrun --standalone --nproc-per-node=2 scripts/trainer_example.py
  torchrun --standalone --nproc-per-node=2 scripts/trainer_example.py --bf16
"""
import os
import math
import argparse
import torch
from torch.utils.data import Dataset

# Trainer requires transformers
try:
    from transformers import (
        GPT2LMHeadModel, GPT2Config,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
except ImportError:
    raise SystemExit(
        "transformers not installed. Run: pip install transformers"
    )


# ─── Synthetic dataset ────────────────────────────────────────────────────────
class SyntheticDataset(Dataset):
    def __init__(self, vocab_size=50257, seq_len=128, n=1000):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n          = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "labels": ids.clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",          type=int,   default=50)
    parser.add_argument("--batch",          type=int,   default=4)
    parser.add_argument("--output-dir",     type=str,   default="/tmp/trainer-output", dest="output_dir")
    parser.add_argument("--bf16",           action="store_true")
    parser.add_argument("--fp16",           action="store_true")
    parser.add_argument("--deepspeed",      type=str,   default=None, help="Path to ds_config.json")
    parser.add_argument("--seq-len",        type=int,   default=128,  dest="seq_len")
    # torchrun injects --local-rank (legacy compat)
    parser.add_argument("--local-rank",     type=int,   default=-1,   dest="local_rank")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    is_main    = (local_rank == 0)

    if is_main:
        rank       = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        gpus       = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        print(f"[TRAINER] HuggingFace Trainer Demo")
        print(f"[TRAINER] RANK={rank}  WORLD_SIZE={world_size}  LOCAL_RANK={local_rank}")
        print(f"[TRAINER] CUDA_VISIBLE_DEVICES={gpus}")
        print(f"[TRAINER] steps={args.steps}  batch_per_gpu={args.batch}")

    # ── Small GPT-2 config for fast demo ──
    config = GPT2Config(
        vocab_size=50257,
        n_positions=args.seq_len,
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)

    if is_main:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[TRAINER] Model: GPT-2 custom ({params:.1f}M params)")

    train_dataset = SyntheticDataset(seq_len=args.seq_len, n=args.steps * args.batch * 4)

    # ── TrainingArguments ──
    # Trainer auto-detects distributed setup from LOCAL_RANK env var.
    # ddp_backend="nccl" is used automatically for multi-GPU.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        logging_steps=10,
        save_steps=args.steps + 1,  # don't save during demo
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=2,
        report_to="none",           # disable wandb/tensorboard for demo
        deepspeed=args.deepspeed,
        # Trainer reads LOCAL_RANK from env — no need to pass it explicitly
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    if is_main:
        print(f"[TRAINER] Starting training...")

    result = trainer.train()

    if is_main:
        print(f"\n[TRAINER] Done!")
        print(f"[TRAINER] Final loss:     {result.training_loss:.4f}")
        print(f"[TRAINER] Total steps:    {result.global_step}")
        print(f"[TRAINER] Runtime:        {result.metrics.get('train_runtime', 0):.1f}s")
        print(f"[TRAINER] Steps/sec:      {result.metrics.get('train_steps_per_second', 0):.2f}")


if __name__ == "__main__":
    main()
