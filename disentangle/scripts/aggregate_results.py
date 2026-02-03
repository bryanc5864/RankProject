"""
Aggregate all evaluation results into a single CSV for analysis.

Usage:
    python scripts/aggregate_results.py \
        --results_dir results/ablation/ \
        --output_file results/ablation/aggregated_results.csv
"""

import argparse
import json
import os

import pandas as pd


def aggregate_results(results_dir: str, output_file: str):
    rows = []

    for run_name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        # Load config for metadata
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        for tier in [1, 2, 3, 4]:
            result_file = os.path.join(run_dir, f"tier{tier}_results.json")
            if not os.path.exists(result_file):
                continue

            with open(result_file) as f:
                results = json.load(f)

            for metric_name, metric_value in results.items():
                if isinstance(metric_value, (int, float)):
                    rows.append({
                        "run_name": run_name,
                        "architecture": config.get("architecture", ""),
                        "condition": config.get("condition", ""),
                        "seed": config.get("seed", ""),
                        "tier": tier,
                        "metric": metric_name,
                        "value": metric_value,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Aggregated {len(df)} metric values from {df['run_name'].nunique()} runs")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.output_file)
