"""Rewrite the J-RUM/exemplar-partitioning HF dataset pickles under the `ep.*` namespace.

The dictionaries on HuggingFace were pickled before the 2026-05-03 cas→ep
rename, so a fresh user running `Dictionary.from_hub` hits
`ModuleNotFoundError: No module named 'cas'`. This script downloads each
blob, deserialises through `_LegacyCASCompatUnpickler`, re-pickles under the
current `ep.discovery.dictionary` module path, verifies the new blob loads
cleanly with a vanilla `pickle.load`, then (optionally) uploads it back.

Usage::

    # Rewrite all blobs into ./repickled/ and verify with vanilla pickle.
    python -m scripts.repickle_hub

    # Then upload (needs a write-scoped HF token via `hf auth login`).
    python -m scripts.repickle_hub --upload
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

import ep  # noqa: F401 — ensures ep.discovery.dictionary is importable for verify
from ep.discovery.dictionary import _LegacyCASCompatUnpickler

REPO_ID = "J-RUM/exemplar-partitioning"

# (model_short, layer, percentile) — matches what's actually on HF as of 2026-05-16.
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
    fname = f"{model_short}_layer{layer}.pkl"
    return f"{subdir}/{fname}"


def repickle_one(model_short: str, layer: int, percentile: int, out_dir: Path) -> Path:
    rel = remote_path(model_short, layer, percentile)
    src = hf_hub_download(repo_id=REPO_ID, filename=rel, repo_type="dataset")
    with open(src, "rb") as f:
        d = _LegacyCASCompatUnpickler(f).load()

    # Confirm the deserialised object is now bound to the ep.* namespace.
    assert type(d).__module__ == "ep.discovery.dictionary", (
        f"unexpected module after compat-load: {type(d).__module__}"
    )
    if d.partitions:
        assert type(d.partitions[0]).__module__ == "ep.discovery.dictionary", (
            f"Partition module after compat-load: {type(d.partitions[0]).__module__}"
        )

    out_path = out_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Verify the new blob loads under vanilla pickle (no compat shim).
    with open(out_path, "rb") as f:
        d2 = pickle.load(f)
    assert len(d2.partitions) == len(d.partitions)
    assert abs(d2.threshold - d.threshold) < 1e-9

    print(
        f"  ok: {rel}  partitions={len(d2.partitions)}  "
        f"θ={d2.threshold:.4f}  size={out_path.stat().st_size:,} B"
    )
    return out_path


def upload(out_dir: Path) -> None:
    api = HfApi()
    for model_short, layer, percentile in BLOBS:
        rel = remote_path(model_short, layer, percentile)
        local = out_dir / rel
        print(f"  upload: {rel} ({local.stat().st_size:,} B)")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Re-pickle {rel} under ep.* namespace (post cas→ep rename)",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("repickled"))
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Push re-pickled blobs back to HF (requires write-scoped auth).",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip the download+repickle step and upload an existing --out-dir.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.upload_only:
        for model_short, layer, percentile in BLOBS:
            repickle_one(model_short, layer, percentile, args.out_dir)
        print(f"\nWrote {len(BLOBS)} re-pickled blobs to {args.out_dir}/")

    if args.upload or args.upload_only:
        upload(args.out_dir)
        print(f"\nUploaded {len(BLOBS)} blobs to {REPO_ID}.")
    else:
        print("\nSkipped upload (pass --upload to push back to HF).")


if __name__ == "__main__":
    main()
