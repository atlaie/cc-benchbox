#!/usr/bin/env python3
"""
build_pairs_500.py — sample 250 pair-rows from ToxicChat for 500 total
prompts.

The existing pairs.json schema (from PHASE3_REFERENCE §2; Phase 2's
50-row "100-pair set"):

    [
      {"pair_id": int, "toxic": <str>, "benign": <str>},
      ...
    ]

Each ROW contains TWO prompts. "Pair count" in the brief's nomenclature
= total prompts = 2 x n_rows. So 250 rows => 500 prompts (250 toxic +
250 benign), matching the brief's convention.

Pairing: toxic and benign within a row are sampled independently and
combined by index. There's no semantic alignment between the two
prompts in a row -- pairing is for the bootstrap's (pair_id, prompt_class)
join, not for content matching.

Usage:
    python build_pairs_500.py \
        --existing-pairs pairs.json \
        --output pairs-500.json \
        --n-rows 250 \
        --seed 42

Dependencies: pip install datasets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REQUIRED_KEYS = {"pair_id", "toxic", "benign"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--existing-pairs", type=Path, default=Path("data/pairs.json"),
                    help="Existing pairs file used to verify schema match")
    ap.add_argument("--output", type=Path, default=Path("data/pairs-500.json"))
    ap.add_argument("--n-rows", type=int, default=250,
                    help="Number of (toxic, benign) rows to emit. "
                         "Total prompts = 2 x n_rows.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-name", default="lmsys/toxic-chat")
    ap.add_argument("--dataset-config", default="toxicchat0124")
    args = ap.parse_args()

    # ---- schema check ----
    if not args.existing_pairs.exists():
        sys.exit(f"FATAL: {args.existing_pairs} not found")
    existing = json.loads(args.existing_pairs.read_text())
    if not existing:
        sys.exit(f"FATAL: {args.existing_pairs} is empty")
    sample_keys = set(existing[0].keys())
    if not REQUIRED_KEYS.issubset(sample_keys):
        sys.exit(
            f"FATAL: expected keys {sorted(REQUIRED_KEYS)} in "
            f"{args.existing_pairs}, got {sorted(sample_keys)}. "
            f"Schema mismatch -- extend this script or update pairs.json."
        )
    extra_keys = sample_keys - REQUIRED_KEYS
    print(f"[schema] required keys present: {sorted(REQUIRED_KEYS)}")
    if extra_keys:
        print(f"[schema] additional keys in existing: {sorted(extra_keys)} "
              f"(will be copied forward from first existing row as defaults)")
    print(f"[existing] N={len(existing)} rows ({2 * len(existing)} prompts)")

    # ---- dataset load ----
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("FATAL: `datasets` library required. `pip install datasets`")

    print(f"[load] loading {args.dataset_name} ({args.dataset_config})...")
    ds = load_dataset(args.dataset_name, args.dataset_config, split="train")

    # toxicchat0124 schema (verified on HF):
    #   user_input: str (the prompt)
    #   toxicity: int (0 = benign, 1 = toxic, human annotation per Lin et al. 2023)
    toxic_pool = [r["user_input"] for r in ds if r["toxicity"] == 1]
    benign_pool = [r["user_input"] for r in ds if r["toxicity"] == 0]
    print(f"[load] dataset has {len(toxic_pool)} toxic, "
          f"{len(benign_pool)} benign user_inputs")

    if len(toxic_pool) < args.n_rows:
        sys.exit(f"FATAL: only {len(toxic_pool)} toxic available, "
                 f"need {args.n_rows}")
    if len(benign_pool) < args.n_rows:
        sys.exit(f"FATAL: only {len(benign_pool)} benign available, "
                 f"need {args.n_rows}")

    # ---- avoid duplicating prompts that are already in pairs.json ----
    # Keeps the new file independent of the original 50-row set in case
    # the user ever wants to combine them.
    existing_toxic = {row["toxic"] for row in existing}
    existing_benign = {row["benign"] for row in existing}
    fresh_toxic = [p for p in toxic_pool if p not in existing_toxic]
    fresh_benign = [p for p in benign_pool if p not in existing_benign]
    print(f"[load] after dedup vs existing: "
          f"{len(fresh_toxic)} toxic, {len(fresh_benign)} benign")
    if len(fresh_toxic) < args.n_rows or len(fresh_benign) < args.n_rows:
        print("[warn] dedup left fewer pool entries than requested; "
              "falling back to including duplicates")
        fresh_toxic = toxic_pool
        fresh_benign = benign_pool

    # ---- sample + pair ----
    random.seed(args.seed)
    random.shuffle(fresh_toxic)
    random.shuffle(fresh_benign)
    toxic = fresh_toxic[:args.n_rows]
    benign = fresh_benign[:args.n_rows]

    pairs = []
    default_extras = {k: v for k, v in existing[0].items()
                      if k not in REQUIRED_KEYS}
    for i in range(args.n_rows):
        row = {"pair_id": i, "toxic": toxic[i], "benign": benign[i]}
        for k, v in default_extras.items():
            row.setdefault(k, v)
        pairs.append(row)

    args.output.write_text(json.dumps(pairs, indent=2))
    print(f"[done] wrote {len(pairs)} rows ({2 * len(pairs)} prompts) "
          f"to {args.output}")
    print(f"       schema: {sorted(pairs[0].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
