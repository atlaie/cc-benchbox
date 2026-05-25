#!/usr/bin/env python3
"""
build_ruler_pairs.py — single-needle NIAH pair builder for the Task C
long-input tok_in sweep.

Produces one JSON pairs file per target tok_in length, schema-compatible
with phase3_vllm_driver.load_pairs(). Each pair contains two distinct
NIAH prompts (different needles) under the existing `toxic` / `benign`
keys; the labels carry no semantic meaning here, they just preserve the
2-per-pair structure phase3_vllm_driver.interleave() expects.

Length control
--------------
Targets the EXACT tokens_in that vLLM will report by applying the GLM-5.1
chat template locally (the same template vLLM applies server-side), then
binary-searching the haystack length to hit target ± 2 tokens. Verified
per-pair counts are recorded in the manifest.

Prefix-cache defeat
-------------------
vLLM ships with prefix caching enabled by default. If every NIAH prompt
shared a long common prefix (chat template wrapper, "<document>\\n", etc.)
the second request onward would get a 15-20 token block-cache hit and
the measured prefill slope would be biased downward, by ~15% at
tok_in=100 (small denominator) and ~0.2% at tok_in=8000 (large
denominator) — exactly the wrong shape for a clean $\\Delta b_1$ fit. To
defeat this each prompt opens with a unique "[Document #<uuid>]" header,
guaranteeing no prefix cache hits past the chat template's first ~8
tokens.

Pair-ID namespace
-----------------
pair_ids start at 11_000 + 10*(target_tok_in_index) so this sweep cannot
collide with the existing pairs.json (0..N) or with future max_tokens
sweep cells.

Usage
-----
  # 1. fetch a haystack (one-time, ~76k words public domain)
  curl -sL https://www.gutenberg.org/cache/epub/84/pg84.txt > ruler_haystack.txt

  # 2. build pairs
  python build_ruler_pairs.py \\
    --haystack-path ruler_haystack.txt \\
    --target-tok-in 100 500 1000 4000 8000 \\
    --pairs-per-target 50 \\
    --output-dir pairs_ruler/ \\
    --tokenizer zai-org/GLM-5.1-FP8 \\
    --seed 42

Outputs:
    pairs_ruler/pairs_t100.json     (50 pairs)
    pairs_ruler/pairs_t500.json
    pairs_ruler/pairs_t1000.json
    pairs_ruler/pairs_t4000.json
    pairs_ruler/pairs_t8000.json
    pairs_ruler/manifest.json       (build metadata + per-target actual tok_in stats)

Exit codes:
  0  success
  2  user error (missing files, bad CLI)
  3  tokenizer / haystack incompatible with smallest target
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Hard imports — clear error if missing.
try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: pip install transformers", file=sys.stderr)
    sys.exit(2)

import numpy as np


# ---------------------------------------------------------------------------
# NIAH templates
# ---------------------------------------------------------------------------
# Standard Greg-Kamradt-style single-needle structure. Needle is inserted
# at depth ~50% of the haystack. The opening "[Document #<uuid>]" line
# defeats vLLM's block-level prefix caching across pairs.

# Curated city pool — short, common, unambiguous tokenizations.
CITIES = [
    "Lisbon", "Toronto", "Kyoto", "Helsinki", "Nairobi", "Quito",
    "Reykjavik", "Wellington", "Tallinn", "Singapore", "Marrakech",
    "Vancouver", "Casablanca", "Stockholm", "Auckland", "Edinburgh",
    "Budapest", "Krakow", "Tbilisi", "Bangkok", "Manila", "Lima",
    "Bogota", "Riga", "Sarajevo", "Reykjanes", "Ljubljana", "Valencia",
    "Porto", "Ankara", "Cape Town", "Casper", "Dakar", "Dublin",
    "Vienna", "Zurich", "Munich", "Brussels", "Geneva", "Antwerp",
    "Cardiff", "Belfast", "Glasgow", "Galway", "Cork", "Bilbao",
    "Granada", "Seville", "Naples", "Florence", "Venice", "Verona",
    "Salzburg", "Bremen", "Bonn", "Aachen", "Leiden", "Utrecht",
    "Bruges", "Ghent", "Tampere", "Turku", "Aarhus", "Odense",
    "Bergen", "Trondheim", "Malmo", "Gothenburg", "Uppsala", "Vilnius",
    "Kaunas", "Tartu", "Riga", "Minsk", "Lviv", "Yerevan", "Baku",
    "Tashkent", "Almaty", "Bishkek", "Dushanbe", "Ashgabat", "Tehran",
    "Isfahan", "Damascus", "Amman", "Beirut", "Jerusalem", "Cairo",
    "Alexandria", "Tunis", "Algiers", "Rabat", "Fez", "Bamako",
    "Dakar", "Abidjan", "Accra", "Lagos", "Kampala", "Kigali",
]


def build_niah_prompt(
    tokenizer,
    haystack_token_ids: np.ndarray,
    target_tok_in: int,
    template_overhead: int,
    doc_id: str,
    city: str,
    password: str,
    rng: np.random.Generator,
    enable_thinking: bool = False,
) -> tuple[str, int]:
    """Build one NIAH prompt whose chat-template-applied length is
    target_tok_in (within ±2). Returns (prompt_text, actual_tok_in).

    Strategy: binary-search the haystack token count such that the total
    chat-template-applied length matches target. Picks a random non-
    overlapping haystack window for each call so two prompts in the same
    pair don't share long suffixes either.
    """
    # Fixed parts of the prompt body (independent of haystack length).
    header = f"[Document #{doc_id}]\n\n<document>\n"
    needle = f"\nThe magic password for {city} is {password}.\n"
    footer = f"\n</document>\n\nQuestion: What is the magic password for {city}?\nAnswer:"

    # Pick a random window in the haystack large enough for the budget.
    # 2x target gives slack for the binary search.
    max_window = min(len(haystack_token_ids) - 1, target_tok_in * 3 + 50)
    if max_window < 5:
        raise RuntimeError(
            f"haystack too short ({len(haystack_token_ids)} tokens) for any window"
        )
    max_start = max(1, len(haystack_token_ids) - max_window)
    window_start = int(rng.integers(0, max_start))
    window_end   = window_start + max_window
    window = haystack_token_ids[window_start:window_end]

    def _try_haystack_len(n_haystack: int) -> tuple[str, int]:
        n_haystack = max(0, min(n_haystack, len(window)))
        # Split haystack ~50/50 around the needle insertion point.
        half = n_haystack // 2
        before_ids = window[:half]
        after_ids  = window[half:n_haystack]
        before_text = tokenizer.decode(before_ids, skip_special_tokens=True) if half > 0 else ""
        after_text  = tokenizer.decode(after_ids, skip_special_tokens=True) if (n_haystack - half) > 0 else ""
        body = f"{header}{before_text}{needle}{after_text}{footer}"
        # Two-step (render to string, then tokenize). transformers 5.x's
        # apply_chat_template(tokenize=True) returns a BatchEncoding dict
        # (`{'input_ids': [...], 'attention_mask': [...]}`) rather than
        # the flat list of ints v4 returned, so `len(ids)` on the dict
        # silently returns 2 instead of the token count. Render-then-
        # tokenize is unambiguous and gives the same token count as
        # tokenize=True. The rendered string already contains the special
        # token literals ([gMASK], <sop>, <|user|>, ...), so
        # add_special_tokens=False is required to avoid double-counting.
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        n = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        return body, n

    # Binary search on haystack token count. Loose-then-tight bracket.
    lo, hi = 0, len(window)
    best_body: Optional[str] = None
    best_n = 0
    best_diff = float("inf")
    for _ in range(40):           # log2(window) iterations are plenty
        mid = (lo + hi) // 2
        body, n = _try_haystack_len(mid)
        diff = abs(n - target_tok_in)
        if diff < best_diff:
            best_diff = diff
            best_body = body
            best_n = n
            if diff <= 2:
                break
        if n < target_tok_in:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break

    assert best_body is not None
    return best_body, best_n


def compute_template_overhead(tokenizer, enable_thinking: bool) -> int:
    """Chat template + role tokens that wrap an empty user message.
    Used as a sanity check and to refuse impossible targets. Two-step
    pattern matches the binary-search path; see build_niah_prompt for
    the transformers 5.x BatchEncoding rationale."""
    s = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return len(tokenizer(s, add_special_tokens=False).input_ids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--haystack-path", type=Path, required=True,
                   help="Plain-text haystack source (e.g. Frankenstein from "
                        "Gutenberg). >= 100k tokens recommended for tok_in=8000.")
    p.add_argument("--target-tok-in", nargs="+", type=int, required=True,
                   help="Target tok_in values (chat-template-applied). "
                        "Recommended: 100 500 1000 4000 8000")
    p.add_argument("--pairs-per-target", type=int, default=50,
                   help="Pairs per target tok_in value. Each pair = 2 prompts "
                        "(toxic + benign labels, both NIAH). Default 50.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tokenizer", default="zai-org/GLM-5.1-FP8",
                   help="HF tokenizer to apply (must match the served model's "
                        "chat template). Default zai-org/GLM-5.1-FP8.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Match driver-side enable_thinking=True. Default off "
                        "(matches phase3_vllm_driver default).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--password-min", type=int, default=10_000)
    p.add_argument("--password-max", type=int, default=99_999)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.haystack_path.exists():
        print(f"ERROR: haystack not found at {args.haystack_path}\n"
              f"  fetch one with:\n"
              f"  curl -sL https://www.gutenberg.org/cache/epub/84/pg84.txt "
              f"> {args.haystack_path}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading tokenizer: {args.tokenizer}")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"  loaded in {time.monotonic() - t0:.1f}s, vocab_size={tokenizer.vocab_size}")

    overhead = compute_template_overhead(tokenizer, args.enable_thinking)
    print(f"  chat-template overhead (empty user msg): {overhead} tokens")
    if min(args.target_tok_in) < overhead + 30:
        print(f"ERROR: smallest target {min(args.target_tok_in)} too small "
              f"(template overhead alone is {overhead} + ~30 for NIAH scaffolding). "
              f"Increase smallest target to >= {overhead + 30}.", file=sys.stderr)
        return 3

    print(f"[2/4] loading haystack: {args.haystack_path}")
    haystack_bytes = args.haystack_path.read_bytes()
    haystack_md5 = hashlib.md5(haystack_bytes).hexdigest()
    haystack_text = haystack_bytes.decode("utf-8", errors="ignore")
    # Tokenize once. Use raw tokenizer (no chat template, no special tokens)
    # — these are filler ids we'll later decode back into text.
    haystack_token_ids = np.asarray(
        tokenizer(haystack_text, add_special_tokens=False).input_ids,
        dtype=np.int64,
    )
    print(f"  haystack: {len(haystack_text)} chars, {len(haystack_token_ids)} tokens, "
          f"md5={haystack_md5[:12]}")

    max_target = max(args.target_tok_in)
    if max_target * 3 > len(haystack_token_ids):
        print(f"WARNING: haystack is only {len(haystack_token_ids)} tokens; "
              f"max target {max_target} needs ~{max_target * 3} for clean "
              f"random windowing. Pairs at max target may share content.",
              file=sys.stderr)

    rng = np.random.default_rng(args.seed)

    print(f"[3/4] building pairs for {len(args.target_tok_in)} target(s) "
          f"× {args.pairs_per_target} pairs/target = "
          f"{len(args.target_tok_in) * args.pairs_per_target} pairs total")

    manifest = {
        "schema_version": "ruler-pairs-v1",
        "tokenizer": args.tokenizer,
        "enable_thinking": args.enable_thinking,
        "template_overhead_tokens": overhead,
        "haystack_path": str(args.haystack_path),
        "haystack_md5": haystack_md5,
        "haystack_n_tokens": int(len(haystack_token_ids)),
        "seed": args.seed,
        "pairs_per_target": args.pairs_per_target,
        "targets": [],
    }

    for target_idx, target in enumerate(args.target_tok_in):
        pair_id_base = 11_000 + 10 * target  # e.g. tok_in=500 -> pair_ids 16000..16049
        pairs: list[dict] = []
        actual_lengths: list[int] = []
        diffs: list[int] = []

        t0 = time.monotonic()
        for i in range(args.pairs_per_target):
            # Two distinct cities + passwords per pair (toxic and benign side).
            c_indices = rng.choice(len(CITIES), size=2, replace=False)
            city_t = CITIES[c_indices[0]]
            city_b = CITIES[c_indices[1]]
            pw_t = int(rng.integers(args.password_min, args.password_max))
            pw_b = int(rng.integers(args.password_min, args.password_max))

            pair_id = pair_id_base + i
            doc_id_t = f"{pair_id:08d}T"  # e.g. 00016000T
            doc_id_b = f"{pair_id:08d}B"

            try:
                prompt_t, n_t = build_niah_prompt(
                    tokenizer, haystack_token_ids, target, overhead,
                    doc_id_t, city_t, pw_t, rng, args.enable_thinking,
                )
                prompt_b, n_b = build_niah_prompt(
                    tokenizer, haystack_token_ids, target, overhead,
                    doc_id_b, city_b, pw_b, rng, args.enable_thinking,
                )
            except RuntimeError as e:
                print(f"  [{target}] pair {i}: {e}", file=sys.stderr)
                continue

            pairs.append({
                "pair_id": pair_id,
                "toxic":  prompt_t,
                "benign": prompt_b,
                # Diagnostic metadata, not consumed by the driver:
                "_tok_in_target":   target,
                "_tok_in_toxic":    n_t,
                "_tok_in_benign":   n_b,
                "_needle_toxic":    {"city": city_t,  "password": str(pw_t)},
                "_needle_benign":   {"city": city_b,  "password": str(pw_b)},
            })
            actual_lengths.extend([n_t, n_b])
            diffs.extend([n_t - target, n_b - target])

        out_path = args.output_dir / f"pairs_t{target}.json"
        out_path.write_text(json.dumps(pairs, indent=2))
        elapsed = time.monotonic() - t0
        arr = np.asarray(actual_lengths)
        diff_arr = np.asarray(diffs)
        stats = {
            "tok_in_target":   target,
            "n_pairs":         len(pairs),
            "pairs_file":      out_path.name,
            "pair_id_range":   [pair_id_base, pair_id_base + len(pairs) - 1],
            "actual_tok_in": {
                "mean":  float(arr.mean()),
                "min":   int(arr.min()),
                "max":   int(arr.max()),
                "stdev": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            },
            "diff_from_target": {
                "mean": float(diff_arr.mean()),
                "min":  int(diff_arr.min()),
                "max":  int(diff_arr.max()),
                "n_within_2":  int((np.abs(diff_arr) <= 2).sum()),
                "n_within_10": int((np.abs(diff_arr) <= 10).sum()),
            },
            "build_seconds": elapsed,
        }
        manifest["targets"].append(stats)
        print(f"  tok_in={target:>5}: {len(pairs)} pairs, "
              f"actual={arr.mean():.1f}±{arr.std(ddof=1):.1f} "
              f"(min={arr.min()}, max={arr.max()}), "
              f"|diff|<=2: {stats['diff_from_target']['n_within_2']}/{len(actual_lengths)}, "
              f"|diff|<=10: {stats['diff_from_target']['n_within_10']}/{len(actual_lengths)}, "
              f"{elapsed:.1f}s")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[4/4] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
