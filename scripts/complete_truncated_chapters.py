#!/usr/bin/env python
"""Surgically complete truncated (finish_reason='length') chapters at a higher token budget.

A thinking-model chapter whose trace exceeds ``--max-tokens`` is cut off (``finish_reason='length'``) — an
INCOMPLETE thinking trace, and often a missing/partial answer. Rather than re-running the whole batch, this
regenerates ONLY those chapters at a larger budget, reusing the **exact original prompt** (retained in
``raw.exchange``, matched by ``hx_exchange_id``), and patches the chapter row + its ``.md`` in place → a
100%-complete-trace corpus with no wasted work. Strict layering: re-completion goes through the gRPC engine.

The engine must be served with ``AEGIR_MAX_MODEL_LEN`` ≥ (max-tokens + prompt) so the longer trace fits.

    just engine-serve   # (AEGIR_MAX_MODEL_LEN 32768, AEGIR_MAX_NUM_SEQS 4)
    uv run --no-sync python scripts/complete_truncated_chapters.py --run <chapters_v0/run_id> --dry-run
    uv run --no-sync python scripts/complete_truncated_chapters.py --run <chapters_v0/run_id> --max-tokens 28000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def prompts_by_exchange_id() -> dict[str, str]:
    """``{exchange_id → user-prompt}`` from raw.exchange (the retained full prompts)."""
    from aegir.hx.catalog import get_exchange_table
    rows = get_exchange_table().scan().to_arrow().to_pylist()
    out: dict[str, str] = {}
    for r in rows:
        msgs = r.get("request_messages")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except (ValueError, TypeError):
                msgs = None
        prompt = ""
        for m in msgs or []:
            if isinstance(m, dict) and m.get("role") == "user":
                prompt = m.get("content", "")
        if r.get("id"):
            out[r["id"]] = prompt
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="chapters run dir (contains chapters.parquet)")
    ap.add_argument("--max-tokens", type=int, default=28000, help="larger budget for the retry (≤ model_len − prompt)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--dry-run", action="store_true", help="report truncated chapters + prompt availability, do not call the engine")
    args = ap.parse_args()

    run = Path(args.run)
    tbl = pq.read_table(run / "chapters.parquet")
    rows = tbl.to_pylist()
    trunc = [r for r in rows if (r.get("finish_reason") or "") == "length"]
    print(f"{len(trunc)}/{len(rows)} chapters truncated (finish_reason='length')")
    if not trunc:
        print("nothing to do — all traces complete ✓")
        return 0

    prompts = prompts_by_exchange_id()
    have = sum(1 for r in trunc if prompts.get(r.get("hx_exchange_id")))
    print(f"prompts recovered from raw.exchange: {have}/{len(trunc)}")
    if args.dry_run:
        for r in trunc:
            p = prompts.get(r.get("hx_exchange_id"), "")
            print(f"  {r['chapter_id'][:10]} topic={r.get('target_topic_id')} prompt={'yes' if p else 'MISSING'} ({len(p)} chars)")
        return 0

    from aegir.engine.client import complete_detailed
    fixed = 0
    for r in trunc:
        cid = r["chapter_id"]
        prompt = prompts.get(r.get("hx_exchange_id"), "")
        if not prompt:
            print(f"  {cid[:10]}: no prompt in raw.exchange — skip")
            continue
        out = complete_detailed(prompt, capability=args.capability, max_tokens=args.max_tokens,
                                temperature=float(r.get("temperature") or 0.4))
        if out.get("finish_reason") != "stop":
            print(f"  {cid[:10]}: still '{out.get('finish_reason')}' at {args.max_tokens} tok — raise further; skip")
            continue
        r["response_text"] = out["text"]
        r["response_reasoning"] = out["reasoning_content"]
        r["response_chars"] = len(out["text"])
        r["prompt_tokens"] = int(out.get("prompt_tokens") or 0)
        r["completion_tokens"] = int(out.get("completion_tokens") or 0)
        r["latency_ms"] = int(out.get("latency_ms") or 0)
        r["finish_reason"] = "stop"
        (run / f"chapter_{cid}.md").write_text(out["text"])
        fixed += 1
        print(f"  {cid[:10]}: completed — {out.get('completion_tokens')} tok, "
              f"{len(out['reasoning_content'])} reasoning chars")

    if fixed:
        pq.write_table(pa.Table.from_pylist(rows, schema=tbl.schema), run / "chapters.parquet")
    print(f"patched {fixed}/{len(trunc)} truncated chapters → chapters.parquet "
          f"(raw.exchange retains the original truncated exchanges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
