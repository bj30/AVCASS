import hashlib
import json
import re
from pathlib import Path

STEMS = ("speech", "music", "sfx", "background")

def load_json(path):
    return json.loads(Path(path).read_text())

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def infer_source_id(dataset, path, root):
    path = Path(path)
    root = Path(root)
    rel = path.relative_to(root).as_posix()
    stem = path.stem
    if dataset == "lrs3":
        return rel.rsplit(".", 1)[0]
    if dataset == "fma":
        match = re.match(r"(?P<track>\d{6})", stem)
        return match.group("track") if match else stem
    if dataset == "vggsound":
        return re.sub(r"(_chunk\d+|_part\d+)$", "", stem)
    if dataset == "fsd50k":
        return re.sub(r"(_chunk\d+|_part\d+)$", "", stem)
    return rel.rsplit(".", 1)[0]

def stable_partition(source_id, seed, test_fraction):
    key = f"{seed}:{source_id}".encode()
    digest = hashlib.sha1(key).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"

def summarize(rows):
    unique_ids = len({row["source_id"] for row in rows})
    return {"items": len(rows), "source_ids": unique_ids}
