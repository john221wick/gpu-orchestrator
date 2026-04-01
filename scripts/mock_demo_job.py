#!/usr/bin/env python3
"""
Small stdlib-only workload for gpu-scheduler mock demos.

It prints the assigned CUDA_VISIBLE_DEVICES value, sleeps for a while,
emits heartbeat logs, and exits cleanly unless --fail is set.
"""

import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock GPU scheduler demo workload")
    parser.add_argument("--name", default="demo-job")
    parser.add_argument("--sleep", type=int, default=20)
    parser.add_argument("--heartbeat", type=int, default=5)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    payload = {
        "name": args.name,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "cuda_visible_devices": visible,
    }

    print("[DEMO] start " + json.dumps(payload, sort_keys=True), flush=True)

    started = time.time()
    remaining = args.sleep
    while remaining > 0:
        print(
            f"[DEMO] heartbeat name={args.name} "
            f"elapsed={int(time.time() - started)}s "
            f"remaining={remaining}s "
            f"cuda={visible}",
            flush=True,
        )
        step = min(args.heartbeat, remaining)
        time.sleep(step)
        remaining -= step

    if args.fail:
        print(f"[DEMO] fail name={args.name}", flush=True)
        return 1

    print(f"[DEMO] done name={args.name} total_runtime={int(time.time() - started)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
