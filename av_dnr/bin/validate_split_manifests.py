#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from _manifest_utils import read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Validate zero source overlap across train/test manifests.")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--include-background", action="store_true")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    stems = ["speech", "music", "sfx"] + (["background"] if args.include_background else [])
    report = {}
    failures = []
    for stem in stems:
        train_rows = read_jsonl(split_dir / "train" / f"{stem}.jsonl")
        test_rows = read_jsonl(split_dir / "test" / f"{stem}.jsonl")
        train_ids = {row["source_id"] for row in train_rows}
        test_ids = {row["source_id"] for row in test_rows}
        overlap = sorted(train_ids & test_ids)
        report[stem] = {
            "train_items": len(train_rows),
            "test_items": len(test_rows),
            "train_source_ids": len(train_ids),
            "test_source_ids": len(test_ids),
            "overlap_count": len(overlap),
        }
        if overlap:
            failures.append((stem, overlap[:20]))
    print(json.dumps(report, indent=2))
    if failures:
        lines = [f"{stem}: {items}" for stem, items in failures]
        raise SystemExit("Split overlap detected:\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
