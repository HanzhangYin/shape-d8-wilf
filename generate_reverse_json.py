#!/usr/bin/env python3
"""Generate reverse_k{n}.json shape-Wilf input data.

The construction has two non-singleton sources:

* the layered classes C_t^pi coming from G_{n-t} direct-summed with a
  suffix pi, after removing earlier suffix layers;
* the exceptional pairs {231 ⊕ pi, 312 ⊕ pi} for pi in S_{n-3}.

The remaining permutations can optionally be emitted as singleton classes.
The default singleton mode reproduces the bundled full files such as
reverse_k5.json, reverse_k6.json, and reverse_k8.json: include residual
singletons except the canonical D8-completion seed 234...n1.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, cast

Pattern = tuple[int, ...]
PatternClass = frozenset[Pattern]


def direct_sum(left: Pattern, right: Pattern) -> Pattern:
    """Return the direct sum left ⊕ right."""
    shift = len(left)
    return tuple(left) + tuple(x + shift for x in right)


def format_pattern(pattern: Pattern) -> str:
    """Format a permutation pattern unambiguously.

    For lengths up to 9, keep the historical compact digit string format
    such as ``2413``.  For lengths 10 and above, use spaces so entries like
    10 are not confused with the two digits 1 and 0.
    """
    if max(pattern, default=0) >= 10:
        return " ".join(str(x) for x in pattern)
    return "".join(str(x) for x in pattern)


def parse_pattern(text: str) -> Pattern:
    text = text.strip()
    if not text:
        raise ValueError("empty pattern")
    if any(ch in text for ch in ", "):
        return tuple(int(x) for x in text.replace(",", " ").split())
    return tuple(int(ch) for ch in text)


def generating_set_g(m: int) -> list[Pattern]:
    """Return G_m = {j...1(j+1)...m : 1 <= j <= m}.

    The slides start from G_2, but G_1={1} is the natural boundary case
    needed for small n.
    """
    if m < 1:
        return []
    return [
        tuple(range(j, 0, -1)) + tuple(range(j + 1, m + 1))
        for j in range(1, m + 1)
    ]


def layer_classes(n: int) -> list[PatternClass]:
    """Return the nonempty C_t^pi layers from the G_m construction."""
    if n < 1:
        raise ValueError("n must be positive")

    earlier: set[Pattern] = set()
    classes: list[PatternClass] = []

    for t in range(1, n):
        current_raw_blocks: list[set[Pattern]] = []
        for pi in permutations(range(1, t + 1)):
            raw = {direct_sum(sigma, pi) for sigma in generating_set_g(n - t)}
            new_material = raw - earlier
            if new_material:
                classes.append(frozenset(new_material))
            current_raw_blocks.append(raw)

        # C_t^pi only subtracts layers with smaller suffix length, so update
        # earlier after all pi in this t-layer have been processed.
        for raw in current_raw_blocks:
            earlier.update(raw)

    return classes


def exceptional_pair_classes(n: int) -> list[PatternClass]:
    """Return classes {231 ⊕ pi, 312 ⊕ pi}, pi in S_{n-3}."""
    if n < 3:
        return []

    left_a = (2, 3, 1)
    left_b = (3, 1, 2)
    return [
        frozenset({direct_sum(left_a, pi), direct_sum(left_b, pi)})
        for pi in permutations(range(1, n - 2))
    ]


def all_permutations(n: int) -> set[Pattern]:
    return set(permutations(range(1, n + 1)))


def omitted_d8_completion_seed(n: int) -> Pattern | None:
    """Return the legacy omitted singleton 23...n1, when it exists."""
    if n < 3:
        return None
    return tuple(range(2, n + 1)) + (1,)


def normalize_class(cls: Iterable[Pattern]) -> PatternClass:
    return frozenset(tuple(p) for p in cls)


def class_sort_key(cls: PatternClass) -> tuple[int, tuple[str, ...]]:
    patterns = tuple(sorted(format_pattern(p) for p in cls))
    # Larger structural classes first, then deterministic lexical order.
    return (-len(cls), patterns)


def record_from_class(cls: PatternClass) -> dict[str, object]:
    patterns = [format_pattern(p) for p in sorted(cls)]
    return {"size": len(patterns), "patterns": patterns}


def build_classes(n: int, *, singletons: str = "omit-d8-seed") -> list[PatternClass]:
    """Build reverse classes for S_n.

    singletons modes:
      * "omit-d8-seed": include residual singleton classes except 23...n1;
      * "all": include every residual singleton class;
      * "none": drop every singleton class, including singleton C_t^pi layers.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if singletons not in {"omit-d8-seed", "all", "none"}:
        raise ValueError("singletons must be one of: omit-d8-seed, all, none")

    classes = layer_classes(n) + exceptional_pair_classes(n)
    used = set().union(*classes) if classes else set()

    if singletons in {"omit-d8-seed", "all"}:
        residual = all_permutations(n) - used
        if singletons == "omit-d8-seed":
            seed = omitted_d8_completion_seed(n)
            if seed is not None:
                residual.discard(seed)
        classes.extend(frozenset({p}) for p in residual)

    if singletons == "none":
        classes = [cls for cls in classes if len(cls) > 1]

    # Defensively deduplicate classes while keeping deterministic output.
    unique = sorted(set(classes), key=class_sort_key)
    return unique


def build_reverse_data(
    n: int,
    *,
    singletons: str = "omit-d8-seed",
) -> dict[str, object]:
    classes = build_classes(n, singletons=singletons)
    return {
        "k": n,
        "record_classes": [record_from_class(cls) for cls in classes],
    }


def normalized_class_sets(payload: dict[str, object]) -> set[frozenset[str]]:
    records = cast(list[dict[str, Any]], payload.get("record_classes", []))
    return {frozenset(record["patterns"]) for record in records}


def summarize(payload: dict[str, object]) -> str:
    records = cast(list[dict[str, Any]], payload["record_classes"])
    pattern_count = sum(record["size"] for record in records)
    nonsingleton = sum(1 for record in records if record["size"] > 1)
    return "\n".join(
        [
            f"k={payload['k']}",
            f"classes: {len(records)}",
            f"patterns: {pattern_count}",
            f"non-singleton classes: {nonsingleton}",
        ]
    )


def main_args(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate reverse_k{n}.json data from G_m layers and exceptional pairs.",
    )
    parser.add_argument("n", type=int, help="permutation length k/n")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to write reverse_k{n}.json",
    )
    parser.add_argument(
        "--singletons",
        choices=["omit-d8-seed", "all", "none"],
        default="omit-d8-seed",
        help=(
            "singleton policy: omit-d8-seed matches bundled full files; "
            "none reproduces sparse non-singleton files such as reverse_k7.json"
        ),
    )
    parser.add_argument(
        "--compare-existing",
        type=Path,
        help="optional existing reverse_k*.json for semantic class-set comparison",
    )
    args = parser.parse_args(argv)

    payload = build_reverse_data(
        args.n,
        singletons=args.singletons,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(summarize(payload))
    print(f"wrote {args.output}")

    if args.compare_existing:
        existing = json.loads(args.compare_existing.read_text())
        same_k = existing.get("k") == payload.get("k")
        same_classes = normalized_class_sets(existing) == normalized_class_sets(payload)
        print("\nComparison")
        print("----------")
        print(f"same k: {same_k}")
        print(f"same class sets: {same_classes}")
        if not (same_k and same_classes):
            return 1

    return 0


def main() -> int:
    return main_args()


if __name__ == "__main__":
    raise SystemExit(main())
