#!/usr/bin/env python3
"""Make a text-only sampled-MTP fixture for `strata-q36 --mtp-sampled-test`.
The output is one CSV token line. The prompt is long enough to exercise several
512-token prompt chunks and the normal Qwen3.6 chat template.
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.frontend import ChatTemplate  # noqa: E402


def fixture(tokenizer, template):
    system = ("Strata sampled speculation uses the same target sampler for verification. " * 220).strip()
    user = "Summarize the practical implications of speculative decoding for this runtime."
    prompt = template.render([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             enable_thinking=False)
    ids = tokenizer.encode(prompt, parse_special=True)
    if len(ids) < 1024:
        raise ValueError("fixture is shorter than 1,024 tokens; increase the system text")
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="GGUF first shard carrying the actual tokenizer/template")
    p.add_argument("--template", help="override only if the HTTP server uses a different exported template")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from strata_tokenizer import Tokenizer
    from gguf_reader import GGUFFile

    tokenizer = Tokenizer.from_gguf(args.model)
    with tempfile.TemporaryDirectory(prefix="q36-sampled-template-") as scratch:
        path = Path(args.template) if args.template else Path(scratch) / "chat_template.jinja"
        if not args.template:
            source = GGUFFile(Path(args.model)).metadata.get("tokenizer.chat_template")
            if not isinstance(source, str):
                raise ValueError("GGUF lacks a chat template; pass --template")
            path.write_text(source, encoding="utf-8")
        ids = fixture(tokenizer, ChatTemplate(path))
    out = Path(args.out)
    out.write_text(",".join(map(str, ids)) + "\n", encoding="utf-8")
    print(f"sampled_mtp_fixture={out} prompt_tokens={len(ids)}")


if __name__ == "__main__":
    main()
