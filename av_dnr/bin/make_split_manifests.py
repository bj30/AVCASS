#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

from _manifest_utils import STEMS, read_jsonl, stable_partition, summarize, write_jsonl

def main():
    parser = argparse.ArgumentParser(description="Create deterministic train/test manifests from source catalogs.")
    parser.add_argument("--catalog-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--include-background", action="store_true")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = ["speech", "music", "sfx"] + (["background"] if args.include_background else [])
    summary = defaultdict(dict)

    for stem in stems:
        catalog = read_jsonl(catalog_dir / f"source_catalog.{stem}.jsonl")
        buckets = {"train": [], "test": []}
        for row in catalog:
            split = stable_partition(row["source_id"], args.seed, args.test_fraction)
            buckets[split].append(row)
        for split, rows in buckets.items():
            write_jsonl(output_dir / split / f"{stem}.jsonl", rows)
            summary[split][stem] = summarize(rows)

    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
