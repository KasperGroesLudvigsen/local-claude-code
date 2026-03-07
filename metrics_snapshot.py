"""
metrics_snapshot.py

Take a before/after snapshot of vLLM metrics for a coding session.
Gives you aggregate stats: total tokens, average throughput, average latency.

Workflow:
    1. python metrics_snapshot.py save --tag before --model qwen3-coder-30b
    2. ... do your coding session in Open WebUI ...
    3. python metrics_snapshot.py save --tag after --model qwen3-coder-30b
    4. python metrics_snapshot.py report --model qwen3-coder-30b

Snapshots are saved to ./snapshots/{model}/{tag}.json
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")


VLLM_URL = "http://localhost:8000"
SNAPSHOT_DIR = Path("./snapshots")


def scrape(base_url: str) -> dict[str, float]:
    resp = requests.get(base_url.rstrip("/") + "/metrics", timeout=10)
    resp.raise_for_status()
    metrics = {}
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name_labels, value_str = parts
        try:
            value = float(value_str)
        except ValueError:
            continue
        bare = re.sub(r"\{[^}]*\}", "", name_labels).strip()
        metrics.setdefault(bare, value)
    return metrics


def save_snapshot(model: str, tag: str, url: str):
    data = scrape(url)
    out = SNAPSHOT_DIR / model
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "tag": tag,
        "metrics": data,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Snapshot saved: {path}")


def load_snapshot(model: str, tag: str) -> dict:
    path = SNAPSHOT_DIR / model / f"{tag}.json"
    if not path.exists():
        sys.exit(f"Snapshot not found: {path}")
    return json.loads(path.read_text())


def d(after: dict, before: dict, key: str) -> float | None:
    a, b = after.get(key), before.get(key)
    if a is None or b is None:
        return None
    return max(a - b, 0.0)


def histo_mean(after: dict, before: dict, metric: str) -> float | None:
    s = d(after, before, f"{metric}_sum")
    c = d(after, before, f"{metric}_count")
    if not s or not c:
        return None
    return s / c


def report(model: str):
    before_snap = load_snapshot(model, "before")
    after_snap  = load_snapshot(model, "after")
    b = before_snap["metrics"]
    a = after_snap["metrics"]

    prompt_tokens     = d(a, b, "vllm:prompt_tokens_total")
    completion_tokens = d(a, b, "vllm:generation_tokens_total")
    total_tokens      = (prompt_tokens or 0) + (completion_tokens or 0)
    num_requests      = d(a, b, "vllm:request_success_total")

    mean_ttft    = histo_mean(a, b, "vllm:time_to_first_token_seconds")
    mean_e2e     = histo_mean(a, b, "vllm:e2e_request_latency_seconds")

    # Prefill throughput: prompt tokens / total TTFT across all requests
    ttft_total   = d(a, b, "vllm:time_to_first_token_seconds_sum")
    prefill_tps  = (prompt_tokens / ttft_total) if (prompt_tokens and ttft_total) else None

    # Decode throughput: completion tokens / (total e2e time - total TTFT)
    # This is more robust than relying on time_per_output_token which may not
    # be emitted in all vLLM configurations.
    e2e_total    = d(a, b, "vllm:e2e_request_latency_seconds_sum")
    decode_time  = (e2e_total - ttft_total) if (e2e_total and ttft_total) else None
    decode_tps   = (completion_tokens / decode_time) if (completion_tokens and decode_time and decode_time > 0) else None

    # Inter-token latency: derive from decode throughput if not directly available
    mean_itl     = histo_mean(a, b, "vllm:time_per_output_token_seconds")
    if mean_itl is None and decode_tps:
        mean_itl = 1 / decode_tps

    t_start = before_snap["timestamp_utc"]
    t_end   = after_snap["timestamp_utc"]

    print(f"\n{'═'*52}")
    print(f"  Session report: {model}")
    print(f"{'═'*52}")
    print(f"  Period:               {t_start[:19]}  →  {t_end[:19]} UTC")
    print(f"  Requests completed:   {int(num_requests or 0)}")
    print(f"{'─'*52}")
    print(f"  Prompt tokens:        {int(prompt_tokens or 0):,}")
    print(f"  Completion tokens:    {int(completion_tokens or 0):,}")
    print(f"  Total tokens:         {int(total_tokens):,}")
    print(f"{'─'*52}")
    print(f"  Prefill throughput:   {f'{prefill_tps:.1f} tok/s' if prefill_tps else 'n/a'}")
    print(f"  Decode throughput:    {f'{decode_tps:.1f} tok/s'  if decode_tps  else 'n/a'}")
    print(f"{'─'*52}")
    print(f"  Mean TTFT:            {f'{mean_ttft*1000:.0f} ms' if mean_ttft else 'n/a'}")
    print(f"  Mean inter-token:     {f'{mean_itl*1000:.1f} ms'  if mean_itl  else 'n/a'}")
    print(f"  Mean e2e latency:     {f'{mean_e2e:.1f} s'        if mean_e2e  else 'n/a'}")
    print(f"{'═'*52}\n")

    # Also write a clean JSON summary for the repo
    summary = {
        "model": model,
        "session_start_utc": t_start,
        "session_end_utc": t_end,
        "requests_completed": int(num_requests or 0),
        "tokens": {
            "prompt": int(prompt_tokens or 0),
            "completion": int(completion_tokens or 0),
            "total": int(total_tokens),
        },
        "throughput": {
            "prefill_tokens_per_sec": round(prefill_tps, 1) if prefill_tps else None,
            "decode_tokens_per_sec":  round(decode_tps, 1)  if decode_tps  else None,
        },
        "latency": {
            "mean_ttft_ms":        round(mean_ttft * 1000, 0) if mean_ttft else None,
            "mean_inter_token_ms": round(mean_itl  * 1000, 1) if mean_itl  else None,
            "mean_e2e_s":          round(mean_e2e, 2)         if mean_e2e  else None,
        },
    }
    summary_path = SNAPSHOT_DIR / model / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save", help="Take a snapshot")
    p_save.add_argument("--tag",   required=True, help="'before' or 'after'")
    p_save.add_argument("--model", required=True)
    p_save.add_argument("--url",   default=VLLM_URL)

    p_rep = sub.add_parser("report", help="Diff before/after and print report")
    p_rep.add_argument("--model", required=True)

    args = parser.parse_args()

    if args.cmd == "save":
        save_snapshot(args.model, args.tag, args.url)
    elif args.cmd == "report":
        report(args.model)


if __name__ == "__main__":
    main()