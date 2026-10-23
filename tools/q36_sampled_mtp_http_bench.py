#!/usr/bin/env python3
"""Compare Qwen3.6 sampled MTP on/off through Strata's HTTP API.

The two arms use the same prompt, sampling fields, seed and output cap.  The
engine arm is selected only by `strata_tune.sampled_mtp`; the server must have
been started with the engine's `--mtp-sampled` option.  Results include the
native decode time and draft counters reported in the response's `timings`.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


ARMS = (("sampled-mtp-off", 0), ("sampled-mtp-on", 1))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8095")
    p.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    p.add_argument("--model", default="strata")
    p.add_argument("--prompt-text")
    p.add_argument("--prompt-file", help="read the prompt text from a file")
    p.add_argument("--repeat", type=int, default=1, help="repeat the prompt text (same text for both arms)")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--frequency-penalty", type=float)
    p.add_argument("--repetition-penalty", type=float)
    p.add_argument("--min-p", type=float)
    p.add_argument("--thinking", choices=("default", "true", "false"), default="default")
    p.add_argument("--json-out", help="write all per-request samples as JSON")
    return p.parse_args()


def request(base_url, api_key, body):
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))


def prompt_text(args):
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            text = f.read()
    elif args.prompt_text:
        text = args.prompt_text
    else:
        text = "Explain, in practical terms, how a GPU keeps a long conversation and why prompt reuse matters."
    return text if args.repeat == 1 else "\n".join([text] * args.repeat)


def body(args, prompt, arm_key):
    sampling = {
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "presence_penalty": args.presence_penalty, "seed": args.seed,
        "strata_tune": {"sampled_mtp": arm_key},
    }
    for name, value in (("frequency_penalty", args.frequency_penalty), ("repetition_penalty", args.repetition_penalty),
                        ("min_p", args.min_p)):
        if value is not None:
            sampling[name] = value
    out = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": args.max_tokens, "stream": False, **sampling}
    if args.thinking != "default":
        out["enable_thinking"] = args.thinking == "true"
    return out


def summarize(samples):
    out = {"requests": len(samples)}
    for name, values in (("wall_ms", "wall_ms"), ("decode_ms", "decode_ms"),
                         ("completion_tokens", "completion_tokens"), ("prompt_tokens", "prompt_tokens"),
                         ("cached_tokens", "cached_tokens"), ("draft_n", "draft_n"),
                         ("draft_n_accepted", "draft_n_accepted"),
                         ("native_decode_tok_s", "native_decode_tok_s")):
        present = [s[name] for s in samples if isinstance(s.get(name), (int, float))]
        out[name] = round(statistics.median(present), 3) if present else None
    if out.get("completion_tokens") and out.get("wall_ms"):
        out["wall_tok_s"] = round(1000.0 * out["completion_tokens"] / out["wall_ms"], 3)
    if out.get("decode_ms") == 0:
        out["native_decode_tok_s"] = None
    return out


def main():
    args = parse_args()
    prompt = prompt_text(args)
    samples = {name: [] for name, _ in ARMS}
    for rep in range(args.warmup + args.reps):
        # alternate the order so engine warmup or cache drift does not favor one arm
        arms = list(ARMS) if rep % 2 == 0 else list(reversed(ARMS))
        for name, flag in arms:
            started = time.perf_counter()
            response = request(args.base_url, args.api_key, body(args, prompt, flag))
            wall_ms = (time.perf_counter() - started) * 1000.0
            usage = response.get("usage") or {}
            timings = response.get("timings") or {}
            details = usage.get("prompt_tokens_details") or {}
            record = {
                "rep": rep - args.warmup, "wall_ms": round(wall_ms, 1),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                "decode_ms": timings.get("predicted_ms"),
                "native_decode_tok_s": timings.get("predicted_per_second"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
            }
            if rep >= args.warmup:
                samples[name].append(record)
            print(f"[{rep:>2}] {name:15s} prompt={record['prompt_tokens']} cached={record['cached_tokens']} "
                  f"out={record['completion_tokens']} finish={record['finish_reason']} wall={record['wall_ms']:.0f}ms "
                  f"decode={record['decode_ms'] or 0:.0f}ms native={record['native_decode_tok_s'] or 0:.1f} tok/s "
                  f"drafts={record['draft_n_accepted'] or 0}/{record['draft_n'] or 0}", flush=True)

    summary = {name: summarize(samples[name]) for name, _ in ARMS}
    print("\nmedian over measured runs")
    for name, _ in ARMS:
        row = summary[name]
        print(f"  {name:15s} wall={row['wall_tok_s'] or 0:.2f} tok/s  native={row['native_decode_tok_s'] or 0:.2f} tok/s  "
              f"decode={row['decode_ms'] or 0:.0f}ms  drafts={row['draft_n_accepted'] or 0}/{row['draft_n'] or 0}")
    on, off = summary["sampled-mtp-on"], summary["sampled-mtp-off"]
    for metric in ("native_decode_tok_s", "wall_tok_s"):
        a, b = on.get(metric), off.get(metric)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
            print(f"  {metric}: sampled MTP on/off speedup={a / b:.3f}, throughput delta={a - b:+.2f} tok/s")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "samples": samples, "summary": summary}, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"benchmark request failed: {exc}", file=sys.stderr)
        sys.exit(1)
