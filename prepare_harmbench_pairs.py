#!/usr/bin/env python3
"""
prepare_harmbench_pairs.py
==========================

Builds a `pairs-harmbench.jsonl` file for the Tier 1C safety eval
cell. Samples HarmBench standard behaviors (Mazeika et al. 2024,
arXiv:2402.04249, centerforaisafety/HarmBench) and writes pairs in
the same schema as `pairs.jsonl` so the existing driver can consume
them without code changes.

Why HarmBench
-------------
- 510 carefully curated behaviors across four functional categories
  (standard, copyright, contextual, multimodal). We use the
  STANDARD subset — self-contained text behaviors with no context
  string or image. Most directly comparable to the GCG / AdvBench
  / TDC2023 lineage.
- The de facto standard in the LLM safety eval literature post-2024;
  AISI, Apollo, and most published red-team work reference it.
- Public, license-permissive, reproducible.

Pairs-schema compatibility
--------------------------
Same approach as `prepare_gsm8k_pairs.py`: pair adjacent HarmBench
behaviors and expose them as toxic_prompt/benign_prompt fields. For
HarmBench BOTH prompts in a pair are harmful behaviors — there's no
benign sanity pair (JailbreakBench provides those; HarmBench does
not). The labels are structural, not semantic; the driver doesn't
know or care.

If a benign sanity-check is desired, append a separate matching
benign set (e.g. from JBB-Behaviors' benign mirrors) — out of scope
for v0 of Tier 1C.

Subset selection
----------------
- `--split test` (default): uses
  `data/behavior_datasets/harmbench_behaviors_text_test.csv` (200
  behaviors, the canonical eval split).
- `--split val`: uses the 41-behavior validation split (used by
  HarmBench's own classifier training/calibration).
- `--split all`: uses the union (`harmbench_behaviors_text_all.csv`,
  400 behaviors total in the standard category).

Only `FunctionalCategory == "standard"` rows are kept — contextual
behaviors require a ContextString field the driver doesn't pass
through, and multimodal needs an image.

Usage
-----
    python prepare_harmbench_pairs.py --n-pairs 25 --split test \\
        --out pairs-harmbench.jsonl

The script fetches the CSV directly from the HarmBench GitHub raw
URL on first run; pass `--csv-path` to use a locally-cached copy.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
from pathlib import Path


HARMBENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/"
)

CSV_BY_SPLIT = {
    "test": "harmbench_behaviors_text_test.csv",
    "val":  "harmbench_behaviors_text_val.csv",
    "all":  "harmbench_behaviors_text_all.csv",
}


def _fetch_harmbench_csv(split: str, cache_dir: Path | None = None) -> str:
    """Fetch HarmBench CSV text, optionally caching to disk."""
    if split not in CSV_BY_SPLIT:
        sys.exit(f"ERROR: --split must be one of {list(CSV_BY_SPLIT)}, "
                 f"got {split!r}")
    filename = CSV_BY_SPLIT[split]
    url = HARMBENCH_RAW_BASE + filename

    if cache_dir is not None:
        cache_path = cache_dir / filename
        if cache_path.exists():
            print(f"Reading cached HarmBench CSV: {cache_path}",
                  file=sys.stderr)
            return cache_path.read_text(encoding="utf-8")

    print(f"Fetching HarmBench CSV from {url} ...", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "tier1c-prep/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / filename
        cache_path.write_text(text, encoding="utf-8")
        print(f"Cached to {cache_path}", file=sys.stderr)

    return text


def _parse_harmbench_csv(csv_text: str) -> list[dict]:
    """
    Parse HarmBench CSV.

    Columns: Behavior, FunctionalCategory, SemanticCategory, Tags,
    ContextString, BehaviorID.

    Returns only rows with FunctionalCategory == "standard"
    (self-contained, no context, no image).
    """
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    standard_rows = [r for r in rows if r.get("FunctionalCategory") == "standard"]
    return standard_rows


def build_pairs(rows: list[dict], n_pairs: int, seed: int = 0) -> list[dict]:
    """Pair adjacent HarmBench behaviors into the driver's schema."""
    n_needed = 2 * n_pairs
    if len(rows) < n_needed:
        sys.exit(
            f"ERROR: HarmBench standard subset has {len(rows)} rows, "
            f"need {n_needed} for {n_pairs} pairs. Try --split all."
        )
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    chosen = shuffled[:n_needed]
    pairs = []
    for i in range(n_pairs):
        r_a = chosen[2 * i]
        r_b = chosen[2 * i + 1]
        pairs.append({
            # pair_id MUST be castable to int — phase3_vllm_driver
            # interleave() does int(p["pair_id"]). Use the row index.
            "pair_id": i,
            # Field names MUST be "toxic" and "benign". For HarmBench
            # both prompts in a pair are harmful behaviors — the
            # toxic/benign labels are structural (driver alternates by
            # them), not semantic claims about harmfulness.
            "toxic": r_a["Behavior"].strip(),
            "benign": r_b["Behavior"].strip(),
            "meta": {
                "source": "harmbench",
                "split": "standard",
                "behavior_id_toxic": r_a.get("BehaviorID", ""),
                "behavior_id_benign": r_b.get("BehaviorID", ""),
                "semantic_category_toxic": r_a.get("SemanticCategory", ""),
                "semantic_category_benign": r_b.get("SemanticCategory", ""),
                "tags_toxic": r_a.get("Tags", ""),
                "tags_benign": r_b.get("Tags", ""),
                "pair_label": f"hb_pair_{i:03d}",  # human-readable id
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
    ap.add_argument("--split", default="test", choices=list(CSV_BY_SPLIT),
                    help="HarmBench split: test (200), val (41), all (400). "
                         "Default test.")
    ap.add_argument("--out", type=Path, default=Path("pairs-harmbench.json"),
                    help="Output JSON path. Driver's load_pairs() expects "
                         "a single JSON array, NOT JSONL. Default "
                         "pairs-harmbench.json.")
    ap.add_argument("--csv-path", type=Path, default=None,
                    help="Local HarmBench CSV path (skip download).")
    ap.add_argument("--cache-dir", type=Path, default=Path(".harmbench_cache"),
                    help="Directory to cache the downloaded CSV "
                         "(default .harmbench_cache).")
    args = ap.parse_args()

    if args.csv_path is not None:
        print(f"Reading local HarmBench CSV: {args.csv_path}", file=sys.stderr)
        csv_text = args.csv_path.read_text(encoding="utf-8")
    else:
        csv_text = _fetch_harmbench_csv(args.split, cache_dir=args.cache_dir)

    rows = _parse_harmbench_csv(csv_text)
    print(f"Parsed {len(rows)} standard behaviors from {args.split} split",
          file=sys.stderr)

    pairs = build_pairs(rows, args.n_pairs, seed=args.seed)
    print(f"Built {len(pairs)} pairs ({2 * len(pairs)} prompts total)",
          file=sys.stderr)

    # JSON array (one document), NOT JSONL. phase3_vllm_driver.load_pairs
    # does json.loads(text) and asserts isinstance(pairs, list).
    args.out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)", file=sys.stderr)

    # Print first pair as sanity check.
    print("\nFirst pair (preview):", file=sys.stderr)
    preview = json.dumps(pairs[0], ensure_ascii=False, indent=2)
    print(preview[:800] + ("..." if len(preview) > 800 else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
