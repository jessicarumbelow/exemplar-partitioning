"""Modal entrypoints for the Taboo and resolution-toy experiments.

Run from the repository root:
    modal run modal/experiments.py::taboo_control --secrets gold
    modal run modal/experiments.py::resolution_separation

Results and model weights go to the Modal volumes named by
``EP_MODAL_RESULTS_VOLUME`` and ``EP_MODAL_MODEL_VOLUME`` (created if
missing). Model downloads need a Modal secret named ``huggingface`` holding
``HF_TOKEN``.
"""

from __future__ import annotations

import os

import modal

app = modal.App("ep-experiments")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "numpy>=1.26", "scipy>=1.12",
        "transformers==4.45.1", "transformer-lens==2.15.4",
        "datasets==3.6.0", "huggingface-hub==0.36.2",
        "scikit-learn>=1.4", "matplotlib", "tqdm",
        "zstandard>=0.22", "peft>=0.13", "safetensors>=0.4",
    )
    .env({"PYTHONPATH": "/root/research/ep"})
    .add_local_dir("ep", remote_path="/root/research/ep/ep")
    .add_local_dir("scripts", remote_path="/root/research/ep/scripts")
)

audit_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy>=1.26", "transformers>=4.50",
                 "accelerate", "huggingface-hub")
    .env({"PYTHONPATH": "/root/research/ep"})
    .add_local_dir("ep", remote_path="/root/research/ep/ep")
    .add_local_dir("scripts", remote_path="/root/research/ep/scripts")
)

results = modal.Volume.from_name(
    os.environ.get("EP_MODAL_RESULTS_VOLUME", "ep-results"), create_if_missing=True)
model_cache = modal.Volume.from_name(
    os.environ.get("EP_MODAL_MODEL_VOLUME", "ep-model-cache"), create_if_missing=True)
worker = dict(
    image=image, gpu="H100", memory=32768,
    volumes={"/vol": results, "/models": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)


def _run(module: str, args: list[str]) -> int:
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/models/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/models/hf_cache"
    os.environ["EP_CALIBRATION_CACHE"] = "/vol/calibration"
    command = [sys.executable, "-m", module, *args]
    print("+", " ".join(command))
    return subprocess.run(command, cwd="/root/research/ep").returncode


@app.function(name="ep-taboo", timeout=7200, **worker)
def _taboo(secret: str, layer: int, percentile: float,
           transcripts_json: str, prompts_file: str, suffix: str) -> dict:
    out = (f"/vol/ep_experiments/taboo/{secret}_L{layer}_p{percentile:g}"
           f"_seed0{suffix}")
    args = ["--secret", secret, "--layer", str(layer), "--percentile",
            str(percentile), "--seed", "0", "--device", "cuda",
            "--output-dir", out]
    if transcripts_json:
        args.extend(["--transcripts-json", transcripts_json])
    if prompts_file:
        args.extend(["--prompts-file", prompts_file])
    rc = _run("scripts.exp_taboo", args)
    results.commit()
    model_cache.commit()
    return {"returncode": rc, "out_dir": out}


@app.local_entrypoint()
def taboo(secrets: str = "gold,chair,smile", layer: int = 20,
          percentiles: str = "12", transcripts_json: str = "",
          prompts_file: str = "", suffix: str = ""):
    jobs = []
    for secret in (x.strip() for x in secrets.split(",") if x.strip()):
        for percentile in (float(x) for x in percentiles.split(",")):
            jobs.append((secret, percentile, _taboo.spawn(
                secret, layer, percentile, transcripts_json, prompts_file,
                suffix)))
    for secret, percentile, handle in jobs:
        print(secret, percentile, handle.get())


@app.function(name="ep-taboo-control", timeout=7200, **worker)
def _taboo_control(secret: str, layer: int, percentile: float,
                   transcripts_json: str, suffix: str,
                   exclude_secret_text: bool) -> dict:
    out = (f"/vol/ep_experiments/taboo/{secret}_control_L{layer}"
           f"_p{percentile:g}_seed0{suffix}")
    args = ["--secret", secret, "--layer", str(layer), "--percentile",
            str(percentile), "--seed", "0", "--device", "cuda",
            "--top-regions", "10", "--transcripts-json", transcripts_json,
            "--output-dir", out]
    if exclude_secret_text:
        args.append("--exclude-secret-text")
    rc = _run("scripts.exp_taboo_control", args)
    results.commit()
    model_cache.commit()
    return {"returncode": rc, "out_dir": out}


@app.local_entrypoint()
def taboo_control(secrets: str = "gold,chair,smile", layer: int = 32,
                  percentile: float = 12, transcripts_layer: int = 20,
                  transcripts_p: float = 12, transcripts_suffix: str = "",
                  suffix: str = "", exclude_secret_text: bool = False):
    jobs = []
    for secret in (x.strip() for x in secrets.split(",") if x.strip()):
        transcript_path = (
            f"/vol/ep_experiments/taboo/{secret}_L{transcripts_layer}"
            f"_p{transcripts_p:g}_seed0{transcripts_suffix}/transcripts.json"
        )
        jobs.append((secret, _taboo_control.spawn(
            secret, layer, percentile, transcript_path, suffix,
            exclude_secret_text)))
    for secret, handle in jobs:
        print(secret, handle.get())


@app.function(name="ep-taboo-inventory", timeout=3600, **worker)
def _taboo_inventory(run_dirs: str, output: str,
                     evaluate_secret: bool = False) -> dict:
    args = ["--run-dirs", run_dirs, "--output", output]
    if evaluate_secret:
        args.append("--evaluate-secret")
    rc = _run("scripts.exp_taboo_inventory", args)
    results.commit()
    return {"returncode": rc, "output": output}


@app.local_entrypoint()
def taboo_inventory(run_dirs: str, output: str,
                    evaluate_secret: bool = False):
    print(_taboo_inventory.remote(run_dirs, output, evaluate_secret))


@app.function(
    name="ep-taboo-audit", timeout=7200, image=audit_image, gpu="L4",
    memory=32768, volumes={"/vol": results, "/models": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def _taboo_audit(run_dir: str, transcripts_json: str,
                 budgets: str = "1,3,5,10") -> dict:
    args = ["--run-dir", run_dir, "--budgets", budgets]
    if transcripts_json:
        args.extend(["--transcripts-json", transcripts_json])
    rc = _run("scripts.exp_taboo_audit", args)
    results.commit()
    return {"returncode": rc, "run_dir": run_dir}


@app.local_entrypoint()
def taboo_audit(run_dirs: str, transcripts_json: str = "",
                budgets: str = "1,3,5,10"):
    handles = [_taboo_audit.spawn(path.strip(), transcripts_json, budgets)
               for path in run_dirs.split(",") if path.strip()]
    for handle in handles:
        print(handle.get())


@app.function(name="ep-resolution-separation", timeout=3600, **worker)
def _resolution_separation(model: str, layer: int, percentiles: str,
                           out_suffix: str) -> dict:
    short = model.split("/")[-1]
    out = (f"/vol/ep_experiments/resolution_separation/{short}_L{layer}"
           f"_seed0{('_' + out_suffix) if out_suffix else ''}")
    rc = _run("scripts.exp_resolution_separation", [
        "--model", model, "--model-short", short, "--layer", str(layer),
        "--percentiles", percentiles, "--seed", "0", "--device", "cuda",
        "--output-root", out])
    results.commit()
    model_cache.commit()
    return {"returncode": rc, "out_dir": out}


@app.local_entrypoint()
def resolution_separation(
    model: str = "google/gemma-2-2b", layer: int = 12,
    percentiles: str = "10,12,14,16,18,20,22,25,30,40,55",
    out_suffix: str = "",
):
    print(_resolution_separation.remote(model, layer, percentiles, out_suffix))
