#!/usr/bin/env python
"""Monitor 1M validation run in real-time and print progress."""

import json
import re
import sys
import time
from pathlib import Path

def tail_log(log_path, n_lines=20):
    """Print last n lines of the log."""
    if not log_path.exists():
        return
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-n_lines:]:
            print(line.rstrip())

def parse_latest_metrics(log_path):
    """Extract the last [learn] line for live metrics."""
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    learn_pattern = r'\[learn\] batch=(\d+) \| tok=([\d,]+) \| k=(\d+) \| loss=([\d.]+) \| r_i=([\d.]+) \| λ=\[eff([\d.]+) dom([\d.]+) rel([\d.]+) div([\d.]+)\] \| tok/s=(\d+)'

    for line in reversed(lines):
        if '[learn]' in line:
            m = re.search(learn_pattern, line)
            if m:
                batch, tok_str, k, loss, r_i, leff, ldom, lrel, ldiv, toks = m.groups()
                return {
                    'batch': int(batch),
                    'tokens': int(tok_str.replace(',', '')),
                    'k': int(k),
                    'loss': float(loss),
                    'r_i': float(r_i),
                    'lambdas': [float(leff), float(ldom), float(lrel), float(ldiv)],
                    'tok_s': int(toks),
                }
    return None

def main():
    root = Path(__file__).resolve().parents[1]
    log_path = root / 'logs' / 'validation_1m.log'

    print("\n" + "="*70)
    print("  MONITORING 1M VALIDATION RUN")
    print("="*70)
    print(f"Log: {log_path}")
    print("Press Ctrl+C to stop monitoring (run continues in background)")
    print("="*70 + "\n")

    last_batch = 0
    start_time = time.time()

    try:
        while True:
            metrics = parse_latest_metrics(log_path)
            if metrics:
                elapsed = time.time() - start_time
                est_total_s = (1_000_000 / max(metrics['tokens'], 1)) * elapsed
                est_remaining_h = (est_total_s - elapsed) / 3600
                pct = (metrics['tokens'] / 1_000_000) * 100

                if metrics['batch'] != last_batch:
                    print(f"[{pct:5.2f}%] batch={metrics['batch']:6d} tok={metrics['tokens']:>10,d} "
                          f"loss={metrics['loss']:.4f} λ=[{metrics['lambdas'][0]:.3f} {metrics['lambdas'][1]:.3f} {metrics['lambdas'][2]:.3f} {metrics['lambdas'][3]:.3f}] "
                          f"K={metrics['k']} tok/s={metrics['tok_s']:3d} r_i={metrics['r_i']:.4f} | ETA {est_remaining_h:.1f}h")
                    last_batch = metrics['batch']

                # Check for completion
                if metrics['tokens'] >= 1_000_000:
                    print("\n✅ RUN COMPLETE")
                    break

            time.sleep(5)  # Poll every 5 sec

    except KeyboardInterrupt:
        print("\n[monitor] stopped (run continues in background)")
        print(f"Resume with: tail -f {log_path}")

if __name__ == '__main__':
    main()
