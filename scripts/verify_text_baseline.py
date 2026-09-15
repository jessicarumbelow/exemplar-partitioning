"""Lexical control for harmful/benign region separation.

Input is a JSON list ordered as all harmful prompts followed by an equal number
of benign prompts. The prompt corpus is not distributed; construct it from the
five benchmarks listed in the paper and an equal-size Alpaca sample.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def harmful_coverage(labels, n_harmful, threshold):
    labels = np.asarray(labels)
    harmful = np.arange(len(labels)) < n_harmful
    covered = 0
    for region in np.unique(labels):
        members = labels == region
        if harmful[members].mean() >= threshold:
            covered += int(harmful[members].sum())
    return covered / n_harmful


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts_json", type=Path)
    parser.add_argument("--clusters", type=int, default=161)
    args = parser.parse_args()
    prompts = json.loads(args.prompts_json.read_text())
    if len(prompts) % 2:
        raise ValueError("expected equal harmful and benign halves")
    n_harmful = len(prompts) // 2
    vectors = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2,
        max_features=20_000, sublinear_tf=True,
    ).fit_transform(prompts)
    labels = MiniBatchKMeans(
        n_clusters=args.clusters, random_state=0, n_init=10, batch_size=512,
    ).fit_predict(vectors)
    for threshold in (0.9, 1.0):
        coverage = harmful_coverage(labels, n_harmful, threshold)
        print(f"harmful coverage at {threshold:.0%} purity: {coverage:.1%}")


if __name__ == "__main__":
    main()
