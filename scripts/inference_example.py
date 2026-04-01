#!/usr/bin/env python3
"""
inference_example.py — Minimal GPU inference server demo

Loads a small model (GPT-2 or MinimalLM fallback) and serves predictions
via a simple HTTP REST endpoint.

Usage (via scheduler):
  gpu-submit --framework python --gpus 1 \\
             --script scripts/inference_example.py \\
             --args "--port 8080 --model gpt2"

Usage (direct):
  python scripts/inference_example.py --port 8080

Endpoints:
  POST /generate   {"prompt": "Hello world", "max_tokens": 50}
  GET  /health     {"status": "ok", "model": "...", "gpu": "..."}
  GET  /metrics    {"requests": N, "avg_latency_ms": X}
"""
import os
import sys
import time
import json
import math
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn as nn

# ── Optional: try to load HuggingFace model ──────────────────────────────────
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ── Optional: vLLM for high-throughput inference ─────────────────────────────
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


# ────────────────────────────────────────────────────────────────────────────
# Minimal inference model (fallback when no HuggingFace)
# ────────────────────────────────────────────────────────────────────────────
class MinimalInferenceModel(nn.Module):
    """Character-level language model for demo without transformers."""
    def __init__(self, vocab_size=256, d_model=256, n_layers=4, seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.layers    = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4,
                                        dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_final  = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len   = seq_len

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.embedding(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        h = self.ln_final(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt_bytes, max_tokens=50, temperature=0.8):
        tokens = list(prompt_bytes[:self.seq_len])
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-self.seq_len:]], dtype=torch.long, device=next(self.parameters()).device)
            logits = self.forward(x)[0, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = logits.argmax().item()
            tokens.append(next_tok)
        return bytes(tokens).decode("latin-1", errors="replace")


# ────────────────────────────────────────────────────────────────────────────
# Global inference engine
# ────────────────────────────────────────────────────────────────────────────
engine    = None
tokenizer = None
device    = None
model_name_display = "unknown"

# Metrics
metrics = {
    "requests": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
}
metrics_lock = threading.Lock()


def load_model(args):
    global engine, tokenizer, device, model_name_display

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFERENCE] Device: {gpu_name}")
    print(f"[INFERENCE] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    # ── vLLM (highest performance) ──
    if HAS_VLLM and args.vllm:
        print(f"[INFERENCE] Loading {args.model} with vLLM...")
        t0 = time.time()
        engine = LLM(model=args.model, dtype="float16")
        model_name_display = f"vllm:{args.model}"
        print(f"[INFERENCE] vLLM model loaded in {time.time()-t0:.1f}s")
        return

    # ── HuggingFace transformers ──
    if HAS_TRANSFORMERS and not args.minimal:
        print(f"[INFERENCE] Loading {args.model} with HuggingFace...")
        t0 = time.time()
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            engine    = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            engine = engine.to(device).eval()
            model_name_display = f"hf:{args.model}"
            params = sum(p.numel() for p in engine.parameters()) / 1e6
            print(f"[INFERENCE] Model loaded: {args.model} ({params:.0f}M params) in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[INFERENCE] Warning: could not load {args.model}: {e}")
            print("[INFERENCE] Falling back to MinimalInferenceModel")
            engine    = None
            tokenizer = None

    # ── Minimal fallback ──
    if engine is None:
        print("[INFERENCE] Loading MinimalInferenceModel...")
        t0 = time.time()
        engine = MinimalInferenceModel(vocab_size=256, d_model=256, n_layers=4).to(device).eval()
        model_name_display = "minimal-char-lm"
        params = sum(p.numel() for p in engine.parameters()) / 1e6
        print(f"[INFERENCE] MinimalInferenceModel loaded ({params:.1f}M params) in {time.time()-t0:.1f}s")


@torch.no_grad()
def generate_text(prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
    global engine, tokenizer

    if HAS_VLLM and hasattr(engine, "generate"):
        # vLLM
        sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs  = engine.generate([prompt], sampling)
        return outputs[0].outputs[0].text

    if tokenizer is not None:
        # HuggingFace
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output_ids = engine.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Minimal char-level model
    return engine.generate(prompt.encode("latin-1", errors="replace"),
                           max_tokens=max_tokens, temperature=temperature)


# ────────────────────────────────────────────────────────────────────────────
# HTTP Handler
# ────────────────────────────────────────────────────────────────────────────
class InferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default access log

    def send_json(self, code: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
            self.send_json(200, {
                "status": "ok",
                "model": model_name_display,
                "gpu": gpu_info or "CPU",
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            })
        elif self.path == "/metrics":
            with metrics_lock:
                n   = metrics["requests"]
                avg = metrics["total_latency_ms"] / n if n > 0 else 0.0
                self.send_json(200, {
                    "requests":        n,
                    "errors":          metrics["errors"],
                    "avg_latency_ms":  round(avg, 2),
                    "model":           model_name_display,
                })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                req       = json.loads(body)
                prompt    = req.get("prompt", "Hello,")
                max_tok   = int(req.get("max_tokens", 50))
                temp      = float(req.get("temperature", 0.8))

                t0        = time.time()
                generated = generate_text(prompt, max_tok, temp)
                latency   = (time.time() - t0) * 1000  # ms

                with metrics_lock:
                    metrics["requests"] += 1
                    metrics["total_latency_ms"] += latency

                print(f"[INFERENCE] /generate  latency={latency:.1f}ms  "
                      f"prompt={prompt[:30]!r}  tokens={max_tok}")
                self.send_json(200, {
                    "generated": generated,
                    "latency_ms": round(latency, 2),
                    "model": model_name_display,
                })
            except Exception as e:
                with metrics_lock:
                    metrics["errors"] += 1
                self.send_json(500, {"error": str(e)})
        else:
            self.send_json(404, {"error": "not found"})


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPU Scheduler Inference Demo")
    parser.add_argument("--port",    type=int,   default=8080,  help="HTTP port")
    parser.add_argument("--host",    type=str,   default="0.0.0.0", help="Bind host")
    parser.add_argument("--model",   type=str,   default="gpt2", help="Model name (HuggingFace id)")
    parser.add_argument("--minimal", action="store_true",        help="Force minimal char-level model")
    parser.add_argument("--vllm",    action="store_true",        help="Use vLLM engine")
    args = parser.parse_args()

    print(f"[INFERENCE] gpu-scheduler inference demo starting")
    load_model(args)

    server = HTTPServer((args.host, args.port), InferenceHandler)
    print(f"[INFERENCE] Listening on http://{args.host}:{args.port}")
    print(f"[INFERENCE] Endpoints:")
    print(f"[INFERENCE]   POST /generate  {{\"prompt\": \"...\", \"max_tokens\": 50}}")
    print(f"[INFERENCE]   GET  /health")
    print(f"[INFERENCE]   GET  /metrics")
    print(f"[INFERENCE] Example: curl -s -X POST http://localhost:{args.port}/generate "
          "-H 'Content-Type: application/json' -d '{\"prompt\": \"Once upon a time\"}'")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[INFERENCE] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
