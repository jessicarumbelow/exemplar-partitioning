"""Fast MIB causal variable localisation test for EP dictionaries.

Runs MCQA on Gemma-2-2b at a single layer with DBM+EP, plus full_vector
and optionally DBM+SAE baselines. Reports IIA per variable.

Usage:
    uv run python scripts/mib_fast_test.py --dictionary outputs/my_run/dictionaries/gemma-2-2b_layer12.pkl
    uv run python scripts/mib_fast_test.py --dictionary outputs/my_run/dictionaries/gemma-2-2b_layer12.pkl --with-sae
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIB_ROOT = REPO_ROOT / "baselines" / "MIB" / "MIB-causal-variable-track"
sys.path.insert(0, str(MIB_ROOT))
sys.path.insert(0, str(MIB_ROOT / "CausalAbstraction"))

import torch

from ep.mib_adapter import make_featurizer


def main():
    parser = argparse.ArgumentParser(description="Fast MIB test for EP dictionaries")
    parser.add_argument("--dictionary", type=str, required=True,
                        help="Path to EP dictionary .pkl")
    parser.add_argument("--layer", type=int, default=12, help="Layer to test (default: 12)")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--with-sae", action="store_true", help="Also run DBM+SAE baseline")
    parser.add_argument("--quick", action="store_true", help="Tiny dataset for smoke-testing")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-features", type=int, default=16, help="DAS/DBM feature dim")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- Load dictionary ---
    dict_path = Path(args.dictionary)
    if not dict_path.exists():
        print(f"ERROR: dictionary not found: {dict_path}")
        return 1
    with dict_path.open("rb") as f:
        dictionary = pickle.load(f)
    n_partitions = (
        len(dictionary.partitions) if hasattr(dictionary, "partitions")
        else len(dictionary)
    )
    print(f"Loaded dictionary: {n_partitions} partitions from {dict_path}")

    # --- MIB imports (after sys.path setup) ---
    from neural.pipeline import LMPipeline
    from experiments.residual_stream_experiment import PatchResidualStream
    from experiments.filter_experiment import FilterExperiment
    from tasks.simple_MCQA.simple_MCQA import (
        get_counterfactual_datasets,
        get_causal_model,
        get_token_positions,
    )

    # --- Load model + task ---
    print(f"Loading {args.model}...")
    pipeline = LMPipeline(args.model, max_new_tokens=1, device=device, dtype=torch.float16)
    pipeline.tokenizer.padding_side = "left"

    causal_model = get_causal_model()
    dataset_size = 10 if args.quick else None
    counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size)

    print("Filtering datasets...")
    checker = lambda output_text, expected: expected in output_text
    filt = FilterExperiment(pipeline, causal_model, checker)
    filtered = filt.filter(counterfactual_datasets, verbose=True, batch_size=args.eval_batch_size)

    token_positions = get_token_positions(pipeline, causal_model)

    train_data = {k: v for k, v in filtered.items() if "train" in k}
    test_data = {k: v for k, v in filtered.items() if "test" in k and "private" not in k}

    layers = [args.layer]
    config = {
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.eval_batch_size,
        "training_epoch": args.epochs,
        "n_features": args.n_features,
        "regularization_coefficient": 0.0,
        "output_scores": False,
    }

    results_dir = args.results_dir or f"mib_results/{dict_path.stem}"
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    all_results = {}

    # --- full_vector baseline ---
    print("\n=== full_vector (swap everything) ===")
    config["method_name"] = "full_vector"
    exp = PatchResidualStream(pipeline, causal_model, layers, token_positions, checker, config=config)
    for target in ["answer_pointer", "answer"]:
        print(f"\n  target: {target}")
        results = exp.perform_interventions(
            test_data, verbose=True,
            target_variables_list=[[target]],
            save_dir=results_dir,
        )
        all_results[f"full_vector/{target}"] = results
    del exp
    gc.collect()
    torch.cuda.empty_cache()

    # --- DBM+EP ---
    print("\n=== DBM+EP (our method) ===")
    config["method_name"] = "DBM+EP"
    ep_featurizer = make_featurizer(dictionary, device=device, dtype=torch.float16)
    featurizers = {}
    for pos in token_positions:
        featurizers[(args.layer, pos.id)] = ep_featurizer

    exp = PatchResidualStream(
        pipeline, causal_model, layers, token_positions, checker,
        featurizers=featurizers, config=config,
    )
    for target in ["answer_pointer", "answer"]:
        print(f"\n  target: {target}")
        md = os.path.join(model_dir, f"DBM+EP_{target}")
        exp.train_interventions(
            train_data, [target], method="DBM", verbose=True, model_dir=md,
        )
        results = exp.perform_interventions(
            test_data, verbose=True,
            target_variables_list=[[target]],
            save_dir=results_dir,
        )
        all_results[f"DBM+EP/{target}"] = results
    del exp
    gc.collect()
    torch.cuda.empty_cache()

    # --- DBM+SAE baseline ---
    if args.with_sae:
        print("\n=== DBM+SAE baseline ===")
        config["method_name"] = "DBM+SAE"
        from sae_lens import SAE

        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{args.layer}/width_16k/canonical",
            device="cpu",
        )

        exp = PatchResidualStream(
            pipeline, causal_model, layers, token_positions, checker, config=config,
        )
        exp.build_SAE_feature_intervention(lambda layer: sae)
        for target in ["answer_pointer", "answer"]:
            print(f"\n  target: {target}")
            md = os.path.join(model_dir, f"DBM+SAE_{target}")
            exp.train_interventions(
                train_data, [target], method="DBM", verbose=True, model_dir=md,
            )
            results = exp.perform_interventions(
                test_data, verbose=True,
                target_variables_list=[[target]],
                save_dir=results_dir,
            )
            all_results[f"DBM+SAE/{target}"] = results
        del exp, sae
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, results in all_results.items():
        print(f"\n{key}:")
        if isinstance(results, dict):
            for dataset_name, dataset_results in results.items():
                if isinstance(dataset_results, dict) and "accuracy" in dataset_results:
                    print(f"  {dataset_name}: IIA = {dataset_results['accuracy']:.3f}")
                elif isinstance(dataset_results, (int, float)):
                    print(f"  {dataset_name}: {dataset_results:.3f}")
                else:
                    print(f"  {dataset_name}: {dataset_results}")
        else:
            print(f"  {results}")

    print(f"\nResults saved to {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
