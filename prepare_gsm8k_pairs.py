#!/usr/bin/env python3
"""
prepare_gsm8k_pairs.py
======================

Builds a `pairs-gsm8k.jsonl` file for the Tier 3G reasoning+thinking
cell. Samples GSM8K (Cobbe et al. 2021, OpenAI's grade-school math
dataset) and writes pairs in the same schema as `pairs.jsonl` so the
existing driver can consume them without code changes.

Why GSM8K
---------
- Well-known, public, reproducible (HuggingFace `openai/gsm8k`,
  `main` config, `test` split — 1,319 problems).
- Chain-of-thought reasoning is *required* to solve correctly, so
  `enable_thinking=true` produces realistic long traces (typical
  trace ~ 200-800 tokens depending on problem complexity, often
  longer on GLM-5.1).
- Difficulty is consistent (grade-school) so realised output
  length distributions are predictable.

Pairs-schema compatibility
--------------------------
The driver alternates rows it reads as "toxic" then "benign" within
each pair. For Tier 3G we just need TWO reasoning prompts per pair;
the toxic/benign labelling is structural (it preserves the driver's
alternation logic) not semantic. We pair adjacent problems and
expose them as toxic_prompt/benign_prompt fields so the existing
driver code path works unchanged.

This is identical to how Phase 2's ToxicChat pair construction
worked: the *labels* are structural; the *prompts* are what the
driver dispatches.

Usage
-----
    pip install datasets        # one-time
    python prepare_gsm8k_pairs.py --n-pairs 25 --out pairs-gsm8k.jsonl

The default seed (`--seed 0`) and stable sort over the HF dataset
make this script deterministic across runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the following math problem "
    "step by step, then state the final numeric answer."
)


def _load_gsm8k_test():
    """
    Load GSM8K test split.

    Tries HuggingFace `datasets` first (recommended). Falls back to
    a clear error message if `datasets` is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "ERROR: `datasets` not installed. Run:\n"
            "    pip install datasets\n"
            "or download GSM8K manually from "
            "https://huggingface.co/datasets/openai/gsm8k and "
            "construct pairs by hand."
        )
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds


def build_pairs(n_pairs: int, seed: int = 0) -> list[dict]:
    """
    Sample 2 * n_pairs problems from GSM8K test, pair adjacently,
    return as a list of pair-records in the driver's schema.
    """
    ds = _load_gsm8k_test()
    n_needed = 2 * n_pairs
    if len(ds) < n_needed:
        sys.exit(
            f"ERROR: GSM8K test has {len(ds)} problems, need {n_needed} "
            f"for {n_pairs} pairs."
        )
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = indices[:n_needed]
    pairs = []
    for i in range(n_pairs):
        idx_a = chosen[2 * i]
        idx_b = chosen[2 * i + 1]
        q_a = ds[idx_a]["question"].strip()
        q_b = ds[idx_b]["question"].strip()
        # Extract reference answer (after "#### ") for downstream
        # correctness analysis. GSM8K answers end with "#### N".
        ans_a = ds[idx_a]["answer"].rsplit("####", 1)[-1].strip()
        ans_b = ds[idx_b]["answer"].rsplit("####", 1)[-1].strip()
        pairs.append({
            # pair_id MUST be castable to int — phase3_vllm_driver
            # interleave() does int(p["pair_id"]). Use the row index.
            "pair_id": i,
            # Field names MUST be "toxic" and "benign" — phase3_vllm_driver
            # load_pairs() validates these exact keys. For Tier 3G both
            # prompts are GSM8K problems; the toxic/benign labels are
            # structural metadata that the driver uses to alternate
            # request order. They are NOT semantic claims about the
            # content.
            "toxic": q_a,
            "benign": q_b,
            # Metadata for downstream analysis. Not consumed by the
            # driver — extra fields are tolerated by load_pairs().
            "meta": {
                "source": "gsm8k",
                "split": "test",
                "hf_idx_toxic": idx_a,
                "hf_idx_benign": idx_b,
                "reference_answer_toxic": ans_a,
                "reference_answer_benign": ans_b,
                "system_prompt": SYSTEM_PROMPT,
                "pair_label": f"gsm8k_pair_{i:03d}",  # human-readable id
            },
        })
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--n-pairs", type=int, default=25,
                    help="Number of pairs to generate (default 25 → "
                         "50 prompts, matches Phase 2/3 N=50 convention).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility (default 0).")
    ap.add_argument("--out", type=Path, default=Path("data/pairs-gsm8k.json"),
                    help="Output JSON path. Driver's load_pairs() expects "
                         "a single JSON array, NOT JSONL. Default "
                         "pairs-gsm8k.json.")
    args = ap.parse_args()

    print(f"Loading GSM8K test split ...", file=sys.stderr)
    pairs = build_pairs(args.n_pairs, seed=args.seed)
    print(f"Sampled {len(pairs)} pairs ({2 * len(pairs)} prompts total)",
          file=sys.stderr)

    # JSON array (one document), NOT JSONL. phase3_vllm_driver.load_pairs
    # does json.loads(text) and asserts isinstance(pairs, list).
    args.out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)", file=sys.stderr)

    # Print first pair as sanity check.
    print("\nFirst pair (preview):", file=sys.stderr)
    print(json.dumps(pairs[0], ensure_ascii=False, indent=2)[:600] + "...",
          file=sys.stderr)


if __name__ == "__main__":
    main()
