#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from _manifest_utils import infer_source_id, summarize, write_jsonl, load_json

def main():
    parser = argparse.ArgumentParser(description="Build source catalogs for AVDnR release splits.")
    parser.add_argument("--config", required=True, help="Path to source_roots JSON config.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated source catalogs.")
    args = parser.parse_args()

    config = load_json(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for stem, spec in config["sources"].items():
        rows = []
        dataset = spec["dataset"]
        pattern = spec.get("glob", "**/*.wav")
        for root in spec["roots"]:
            root_path = Path(root)
            matches = sorted(root_path.glob(pattern))
            for path in matches:
                if not path.is_file():
                    continue
                rows.append({
                    "dataset": dataset,
                    "stem": stem,
                    "root": str(root_path),
                    "path": str(path),
                    "relpath": path.relative_to(root_path).as_posix(),
                    "source_id": infer_source_id(dataset, path, root_path),
                })
        if not rows:
            raise SystemExit(f"No files matched for stem={stem}. Check config roots for {dataset}.")
        write_jsonl(output_dir / f"source_catalog.{stem}.jsonl", rows)
        summary[stem] = summarize(rows)

    (output_dir / "source_catalog.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
