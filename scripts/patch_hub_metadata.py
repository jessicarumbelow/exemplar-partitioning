"""Patch `wandb_project: cas` → `wandb_project: ep` in the HF metadata JSONs.

Each prebuilt dictionary on `J-RUM/exemplar-partitioning` ships with a
metadata.json describing the build config. Those configs were written when
the package was named `cas` and still log `"wandb_project": "cas"`. The
field is cosmetic (no code reads it on load) but inconsistent with the
public package name, so we rewrite it.

Usage::

    python -m scripts.patch_hub_metadata               # download + rewrite locally
    python -m scripts.patch_hub_metadata --upload      # also push back to HF
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "J-RUM/exemplar-partitioning"

# (model_short, layer, percentile)
BLOBS = [
    ("gemma-2-2b", 12, 1),
    ("gemma-2-2b", 12, 2),
    ("gemma-2-2b", 12, 4),
    ("gemma-2-2b", 12, 8),
    ("gemma-2-2b", 12, 10),
    ("gemma-2-2b", 20, 10),
    ("gemma-2-2b-it", 4, 4),
    ("gemma-2-2b-it", 12, 10),
    ("gemma-2-2b-it", 20, 1),
    ("gemma-2-2b-it", 20, 2),
    ("gemma-2-2b-it", 20, 4),
    ("gemma-2-2b-it", 20, 8),
    ("gemma-2-2b-it", 20, 10),
]


def remote_path(model_short: str, layer: int, percentile: int) -> str:
    subdir = f"{model_short}_L{layer}_p{int(percentile)}"
    fname = f"{model_short}_layer{layer}_metadata.json"
    return f"{subdir}/{fname}"


def patch_one(model_short: str, layer: int, percentile: int, out_dir: Path) -> tuple[Path, bool]:
    rel = remote_path(model_short, layer, percentile)
    src = hf_hub_download(repo_id=REPO_ID, filename=rel, repo_type="dataset")
    with open(src) as f:
        meta = json.load(f)

    config = meta.get("config", {})
    changed = False
    if config.get("wandb_project") == "cas":
        config["wandb_project"] = "ep"
        changed = True

    out_path = out_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {'patched' if changed else 'no-change'}: {rel}")
    return out_path, changed


def upload(out_dir: Path, changed_paths: list[str]) -> None:
    api = HfApi()
    for rel in changed_paths:
        local = out_dir / rel
        print(f"  upload: {rel}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Rename wandb_project cas→ep in {rel}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("repickled"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    changed_paths: list[str] = []
    for model_short, layer, percentile in BLOBS:
        _, changed = patch_one(model_short, layer, percentile, args.out_dir)
        if changed:
            changed_paths.append(remote_path(model_short, layer, percentile))

    print(f"\n{len(changed_paths)} files needed patching.")
    if args.upload and changed_paths:
        upload(args.out_dir, changed_paths)
        print(f"Uploaded {len(changed_paths)} patched metadata files.")
    elif not args.upload:
        print("Skipped upload (pass --upload to push to HF).")


if __name__ == "__main__":
    main()
