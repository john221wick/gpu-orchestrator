#!/usr/bin/env python3
"""
finetune_example.py — Minimal DeepSpeed/torchrun fine-tuning demo

Trains GPT-2 (small, ~117M params) on a synthetic text dataset for a few steps.
Designed to complete in 2-3 minutes for demonstration purposes.

Usage (via scheduler):
  gpu-submit --framework deepspeed --gpus 2 --peer \\
             --script scripts/finetune_example.py \\
             --args "--steps 100 --batch 4"

Usage (direct):
  deepspeed --num_gpus=2 scripts/finetune_example.py --steps 100
  torchrun --nproc_per_node=2 scripts/finetune_example.py --steps 100
"""
import os
import sys
import time
import argparse
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

# ── Try to import DeepSpeed (optional) ──────────────────────────────────────
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

# ── Try to import transformers GPT-2 ─────────────────────────────────────────
try:
    from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ────────────────────────────────────────────────────────────────────────────
# Synthetic dataset: random token sequences for language modelling
# ────────────────────────────────────────────────────────────────────────────
class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size=50257, seq_len=128, num_samples=2000):
        self.vocab_size  = vocab_size
        self.seq_len     = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Reproducible random token IDs
        torch.manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens, "labels": tokens.clone()}


# ────────────────────────────────────────────────────────────────────────────
# Minimal GPT-2-sized transformer (no external deps fallback)
# ────────────────────────────────────────────────────────────────────────────
class MinimalTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1    = nn.LayerNorm(d_model)
        self.ln2    = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.ln1(x + self.drop(attn_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class MinimalLM(nn.Module):
    """Lightweight language model for demo when transformers not installed."""
    def __init__(self, vocab_size=50257, d_model=256, n_layers=4, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.blocks    = nn.Sequential(*[MinimalTransformerBlock(d_model) for _ in range(n_layers)])
        self.ln_final  = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embedding(input_ids) + self.pos_emb(pos)
        x    = self.blocks(x)
        x    = self.ln_final(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return {"loss": loss, "logits": logits}


# ────────────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────────────
def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))

def is_main():
    return get_rank() == 0


def train(args):
    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main():
        world = get_world_size()
        gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "CPU"
        print(f"[FINETUNE] Starting training | world_size={world} | device={gpu_name}")
        print(f"[FINETUNE] Steps={args.steps} | batch={args.batch} | lr={args.lr}")

    # ── Model ──
    if HAS_TRANSFORMERS and not args.minimal:
        config = GPT2Config(
            vocab_size=50257,
            n_positions=args.seq_len,
            n_embd=256,    # small for demo speed
            n_layer=4,
            n_head=4,
        )
        model = GPT2LMHeadModel(config)
        if is_main():
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[FINETUNE] Model: GPT-2 custom ({params:.1f}M params)")
    else:
        model = MinimalLM(vocab_size=50257, seq_len=args.seq_len)
        if is_main():
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[FINETUNE] Model: MinimalLM ({params:.1f}M params)")

    model = model.to(device)

    # ── Dataset / DataLoader ──
    dataset = SyntheticTextDataset(seq_len=args.seq_len, num_samples=args.steps * args.batch * 2)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=get_world_size(), rank=get_rank()
    ) if get_world_size() > 1 else None

    loader = DataLoader(
        dataset, batch_size=args.batch,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    # ── DeepSpeed or DDP ──
    if HAS_DEEPSPEED and not args.no_deepspeed:
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": args.lr, "weight_decay": 0.01}
            },
            "fp16": {"enabled": torch.cuda.is_available()},
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 1.0,
            "wall_clock_breakdown": False,
        }
        model, optimizer, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer,
            config=ds_config, dist_init_required=False
        )
        if is_main():
            print("[FINETUNE] Using DeepSpeed ZeRO Stage 2")
    else:
        if get_world_size() > 1:
            model = DDP(model, device_ids=[local_rank])
        if is_main():
            print("[FINETUNE] Using standard DDP")

    # ── Training loop ──
    model.train()
    step = 0
    total_loss = 0.0
    t0 = time.time()

    for batch in loader:
        if step >= args.steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        if HAS_DEEPSPEED and not args.no_deepspeed and isinstance(model, deepspeed.DeepSpeedEngine):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            model.backward(loss)
            model.step()
        else:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler_lr.step()

        total_loss += loss.item()
        step += 1

        if is_main() and (step % 10 == 0 or step == 1):
            avg_loss = total_loss / step
            elapsed  = time.time() - t0
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
            print(f"[FINETUNE] Step {step:4d}/{args.steps}  "
                  f"loss={avg_loss:.4f}  ppl={perplexity:.1f}  "
                  f"speed={steps_per_sec:.1f} steps/s  "
                  f"elapsed={elapsed:.0f}s")

    if is_main():
        total_time = time.time() - t0
        avg_loss   = total_loss / max(step, 1)
        print(f"\n[FINETUNE] Training complete!")
        print(f"[FINETUNE] Steps: {step} | Avg loss: {avg_loss:.4f} | "
              f"Time: {total_time:.1f}s | "
              f"Speed: {step/total_time:.2f} steps/s")

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="GPU Scheduler Fine-tuning Demo")
    parser.add_argument("--steps",       type=int,   default=100,    help="Training steps")
    parser.add_argument("--batch",       type=int,   default=4,      help="Batch size per GPU")
    parser.add_argument("--lr",          type=float, default=1e-4,   help="Learning rate")
    parser.add_argument("--seq-len",     type=int,   default=128,    help="Sequence length", dest="seq_len")
    parser.add_argument("--minimal",     action="store_true",        help="Force MinimalLM even if transformers installed")
    parser.add_argument("--no-deepspeed",action="store_true",        help="Use DDP instead of DeepSpeed", dest="no_deepspeed")

    # DeepSpeed launcher injects --local_rank
    parser.add_argument("--local_rank",  type=int, default=-1)

    args = parser.parse_args()

    if torch.cuda.is_available():
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        if is_main():
            print(f"[FINETUNE] CUDA_VISIBLE_DEVICES={gpus}")
            print(f"[FINETUNE] Available GPUs: {torch.cuda.device_count()}")

    train(args)


if __name__ == "__main__":
    main()
