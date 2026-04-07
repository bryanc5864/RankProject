#!/usr/bin/env python3
"""
Parallel Noise-Resistant Training Campaign

Distributes experiments across multiple GPUs for efficient execution.
"""

import subprocess
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Experiment:
    name: str
    model: str
    loss: str
    seed: int
    extra_args: List[str]

# All experiments organized by phase
PHASE1_EXPERIMENTS = [
    # Rank Stability (RS1-RS9)
    Experiment("RS1_bilstm_rs", "dream_rnn_single", "rank_stability", 42, ["--noise_k", "1.0"]),
    Experiment("RS2_bilstm_rs", "dream_rnn_single", "rank_stability", 123, ["--noise_k", "1.0"]),
    Experiment("RS3_bilstm_rs", "dream_rnn_single", "rank_stability", 456, ["--noise_k", "1.0"]),
    Experiment("RS4_factorized_rs", "factorized", "rank_stability", 42, ["--noise_k", "1.0"]),
    Experiment("RS5_factorized_rs", "factorized", "rank_stability", 123, ["--noise_k", "1.0"]),
    Experiment("RS6_factorized_rs", "factorized", "rank_stability", 456, ["--noise_k", "1.0"]),
    Experiment("RS7_bilstm_rs_k2", "dream_rnn_single", "rank_stability", 42, ["--noise_k", "2.0"]),
    Experiment("RS8_bilstm_rs_k05", "dream_rnn_single", "rank_stability", 42, ["--noise_k", "0.5"]),
    Experiment("RS9_factorized_vib_rs", "factorized_vib", "rank_stability", 42, ["--noise_k", "1.0", "--vib_beta", "0.01"]),

    # Distributional Head (DH1-DH9)
    Experiment("DH1_distributional", "dream_rnn_distributional", "distributional", 42, ["--lambda_var", "1.0"]),
    Experiment("DH2_distributional", "dream_rnn_distributional", "distributional", 123, ["--lambda_var", "1.0"]),
    Experiment("DH3_distributional", "dream_rnn_distributional", "distributional", 456, ["--lambda_var", "1.0"]),
    Experiment("DH4_heteroscedastic", "dream_rnn_distributional", "heteroscedastic_distributional", 42, ["--lambda_var", "0.5"]),
    Experiment("DH5_heteroscedastic", "dream_rnn_distributional", "heteroscedastic_distributional", 123, ["--lambda_var", "0.5"]),
    Experiment("DH6_heteroscedastic", "dream_rnn_distributional", "heteroscedastic_distributional", 456, ["--lambda_var", "0.5"]),
    Experiment("DH7_distributional_dual", "dream_rnn_distributional_dual", "distributional", 42, ["--lambda_var", "1.0"]),
    Experiment("DH8_distributional_lv2", "dream_rnn_distributional", "distributional", 42, ["--lambda_var", "2.0"]),
    Experiment("DH9_distributional_lv05", "dream_rnn_distributional", "distributional", 42, ["--lambda_var", "0.5"]),

    # Contrastive Anchor (CA1-CA9)
    Experiment("CA1_contrastive", "dream_rnn_single", "contrastive_anchor", 42, ["--temperature", "0.1"]),
    Experiment("CA2_contrastive", "dream_rnn_single", "contrastive_anchor", 123, ["--temperature", "0.1"]),
    Experiment("CA3_contrastive", "dream_rnn_single", "contrastive_anchor", 456, ["--temperature", "0.1"]),
    Experiment("CA4_triplet", "dream_rnn_single", "triplet_anchor", 42, []),
    Experiment("CA5_triplet", "dream_rnn_single", "triplet_anchor", 123, []),
    Experiment("CA6_triplet", "dream_rnn_single", "triplet_anchor", 456, []),
    Experiment("CA7_factorized_contrastive", "factorized", "contrastive_anchor", 42, ["--temperature", "0.1"]),
    Experiment("CA8_contrastive_t05", "dream_rnn_single", "contrastive_anchor", 42, ["--temperature", "0.05"]),
    Experiment("CA9_contrastive_t02", "dream_rnn_single", "contrastive_anchor", 42, ["--temperature", "0.2"]),

    # Noise Gated (NG1-NG9)
    Experiment("NG1_noise_gated", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.3", "--beta", "0.1", "--noise_k", "1.0"]),
    Experiment("NG2_noise_gated", "dream_rnn_distributional", "noise_gated", 123, ["--alpha", "0.3", "--beta", "0.1", "--noise_k", "1.0"]),
    Experiment("NG3_noise_gated", "dream_rnn_distributional", "noise_gated", 456, ["--alpha", "0.3", "--beta", "0.1", "--noise_k", "1.0"]),
    Experiment("NG4_adaptive_ng", "dream_rnn_distributional", "adaptive_noise_gated", 42, ["--alpha", "0.5", "--beta", "0.1", "--warmup_epochs", "10"]),
    Experiment("NG5_adaptive_ng", "dream_rnn_distributional", "adaptive_noise_gated", 123, ["--alpha", "0.5", "--beta", "0.1", "--warmup_epochs", "10"]),
    Experiment("NG6_adaptive_ng", "dream_rnn_distributional", "adaptive_noise_gated", 456, ["--alpha", "0.5", "--beta", "0.1", "--warmup_epochs", "10"]),
    Experiment("NG7_noise_gated_mse", "dream_rnn_single", "noise_gated_mse", 42, ["--alpha", "0.3", "--noise_k", "1.0"]),
    Experiment("NG8_ng_alpha05", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.5", "--beta", "0.1"]),
    Experiment("NG9_ng_alpha01", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.1", "--beta", "0.1"]),
]

PHASE2_EXPERIMENTS = [
    # Quantile Stratified (QS1-QS6)
    Experiment("QS1_quantile", "dream_rnn_single", "mse", 42, ["--sampler", "quantile_stratified", "--n_quantiles", "10"]),
    Experiment("QS2_quantile", "dream_rnn_single", "mse", 123, ["--sampler", "quantile_stratified", "--n_quantiles", "10"]),
    Experiment("QS3_quantile", "dream_rnn_single", "mse", 456, ["--sampler", "quantile_stratified", "--n_quantiles", "10"]),
    Experiment("QS4_quantile_noise", "dream_rnn_single", "mse", 42, ["--sampler", "quantile_noise_weighted", "--n_quantiles", "10", "--noise_weight", "0.5"]),
    Experiment("QS5_quantile_noise", "dream_rnn_single", "mse", 123, ["--sampler", "quantile_noise_weighted", "--n_quantiles", "10", "--noise_weight", "0.5"]),
    Experiment("QS6_quantile_noise", "dream_rnn_single", "mse", 456, ["--sampler", "quantile_noise_weighted", "--n_quantiles", "10", "--noise_weight", "0.5"]),

    # Quantile Curriculum (QC1-QC6)
    Experiment("QC1_curriculum", "dream_rnn_single", "mse", 42, ["--sampler", "quantile_stratified", "--n_quantiles", "5", "--quantile_curriculum"]),
    Experiment("QC2_curriculum", "dream_rnn_single", "mse", 123, ["--sampler", "quantile_stratified", "--n_quantiles", "5", "--quantile_curriculum"]),
    Experiment("QC3_curriculum", "dream_rnn_single", "mse", 456, ["--sampler", "quantile_stratified", "--n_quantiles", "5", "--quantile_curriculum"]),
    Experiment("QC4_factorized_curriculum", "factorized", "mse", 42, ["--sampler", "quantile_stratified", "--n_quantiles", "5", "--quantile_curriculum"]),
    Experiment("QC5_factorized_curriculum", "factorized", "mse", 123, ["--sampler", "quantile_stratified", "--n_quantiles", "5", "--quantile_curriculum"]),
    Experiment("QC6_factorized_curriculum", "factorized", "mse", 456, ["--sampler", "quantile_stratified", "--n_quantiles", "5", "--quantile_curriculum"]),

    # Hard Negative (HN1-HN6)
    Experiment("HN1_hard_negative", "dream_rnn_single", "mse", 42, ["--sampler", "hard_negative", "--temperature", "1.0"]),
    Experiment("HN2_hard_negative", "dream_rnn_single", "mse", 123, ["--sampler", "hard_negative", "--temperature", "1.0"]),
    Experiment("HN3_hard_negative", "dream_rnn_single", "mse", 456, ["--sampler", "hard_negative", "--temperature", "1.0"]),
    Experiment("HN4_factorized_hn", "factorized", "mse", 42, ["--sampler", "hard_negative", "--temperature", "1.0"]),
    Experiment("HN5_hn_temp2", "dream_rnn_single", "mse", 42, ["--sampler", "hard_negative", "--temperature", "2.0"]),
    Experiment("HN6_hn_temp05", "dream_rnn_single", "mse", 42, ["--sampler", "hard_negative", "--temperature", "0.5"]),
]

PHASE3_EXPERIMENTS = [
    # Factorized Encoder variants (FE1-FE9)
    Experiment("FE1_factorized", "factorized", "mse", 42, []),
    Experiment("FE2_factorized", "factorized", "mse", 123, []),
    Experiment("FE3_factorized", "factorized", "mse", 456, []),
    Experiment("FE4_factorized_vib", "factorized_vib", "mse", 42, ["--vib_beta", "0.01"]),
    Experiment("FE5_factorized_vib", "factorized_vib", "mse", 123, ["--vib_beta", "0.01"]),
    Experiment("FE6_factorized_vib", "factorized_vib", "mse", 456, ["--vib_beta", "0.01"]),
    Experiment("FE7_factorized_gc", "factorized_gc_adv", "mse", 42, ["--gc_bins", "10"]),
    Experiment("FE8_factorized_gc", "factorized_gc_adv", "mse", 123, ["--gc_bins", "10"]),
    Experiment("FE9_factorized_full", "factorized_full", "mse", 42, ["--vib_beta", "0.01", "--gc_bins", "10"]),
]

PHASE4_EXPERIMENTS = [
    # Ablations
    Experiment("ABL1_ng_a01", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.1", "--beta", "0.1"]),
    Experiment("ABL2_ng_a02", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.2", "--beta", "0.1"]),
    Experiment("ABL3_ng_a04", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.4", "--beta", "0.1"]),
    Experiment("ABL4_ng_a05", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.5", "--beta", "0.1"]),
    Experiment("ABL5_ng_b005", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.3", "--beta", "0.05"]),
    Experiment("ABL6_ng_b02", "dream_rnn_distributional", "noise_gated", 42, ["--alpha", "0.3", "--beta", "0.2"]),
    Experiment("ABL7_rs_k15", "dream_rnn_single", "rank_stability", 42, ["--noise_k", "1.5"]),
    Experiment("ABL8_rs_k25", "dream_rnn_single", "rank_stability", 42, ["--noise_k", "2.5"]),

    # Best Combinations
    Experiment("BEST1_ng_qs", "dream_rnn_distributional", "noise_gated", 42,
               ["--alpha", "0.3", "--beta", "0.1", "--sampler", "quantile_stratified", "--n_quantiles", "10"]),
    Experiment("BEST2_ng_qsn", "dream_rnn_distributional", "noise_gated", 42,
               ["--alpha", "0.3", "--beta", "0.1", "--sampler", "quantile_noise_weighted", "--n_quantiles", "10", "--noise_weight", "0.5"]),
    Experiment("BEST3_ng_hn", "dream_rnn_distributional", "noise_gated", 42,
               ["--alpha", "0.3", "--beta", "0.1", "--sampler", "hard_negative"]),
    Experiment("BEST4_full_ng", "factorized_full", "noise_gated", 42,
               ["--alpha", "0.3", "--beta", "0.1", "--vib_beta", "0.01", "--gc_bins", "10", "--sampler", "quantile_stratified", "--n_quantiles", "10"]),
]

ALL_PHASES = {
    1: PHASE1_EXPERIMENTS,
    2: PHASE2_EXPERIMENTS,
    3: PHASE3_EXPERIMENTS,
    4: PHASE4_EXPERIMENTS,
}


def run_experiment(exp: Experiment, gpu_id: int, data_path: str, output_dir: str,
                   epochs: int = 80, batch_size: int = 1024) -> dict:
    """Run a single experiment on specified GPU."""

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp.name}.log"

    # Set up environment with CUDA and GCC library paths
    env = os.environ.copy()
    lib_paths = [
        "/home/bcheng/.conda/envs/phantom/lib",  # For GLIBCXX_3.4.30
        "/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cusparselt/lib",
        "/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cublas/lib",
        "/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
        "/home/bcheng/.conda/envs/physiformer/lib/python3.10/site-packages/nvidia/cudnn/lib",
    ]
    existing_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + ":" + existing_ld_path

    cmd = [
        sys.executable, "scripts/train_noise_resistant.py",
        "--data", data_path,
        "--out", output_dir,
        "--experiment", exp.name,
        "--model", exp.model,
        "--loss", exp.loss,
        "--seed", str(exp.seed),
        "--gpu", str(gpu_id),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ] + exp.extra_args

    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=7200,  # 2 hour timeout
                cwd=str(Path(__file__).parent.parent),
                env=env
            )

        elapsed = time.time() - start_time
        success = result.returncode == 0

        return {
            'name': exp.name,
            'gpu': gpu_id,
            'success': success,
            'elapsed': elapsed,
            'log': str(log_file)
        }

    except subprocess.TimeoutExpired:
        return {
            'name': exp.name,
            'gpu': gpu_id,
            'success': False,
            'elapsed': 7200,
            'error': 'timeout',
            'log': str(log_file)
        }
    except Exception as e:
        return {
            'name': exp.name,
            'gpu': gpu_id,
            'success': False,
            'elapsed': time.time() - start_time,
            'error': str(e),
            'log': str(log_file)
        }


def run_parallel_campaign(experiments: List[Experiment], gpu_ids: List[int],
                          data_path: str, output_dir: str,
                          epochs: int = 80, batch_size: int = 1024):
    """Run experiments in parallel across multiple GPUs."""

    n_gpus = len(gpu_ids)
    n_experiments = len(experiments)

    print(f"Running {n_experiments} experiments across {n_gpus} GPUs: {gpu_ids}")
    print("=" * 60)

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        # Submit all experiments, cycling through GPUs
        futures = {}
        for i, exp in enumerate(experiments):
            gpu_id = gpu_ids[i % n_gpus]
            future = executor.submit(
                run_experiment, exp, gpu_id, data_path, output_dir, epochs, batch_size
            )
            futures[future] = exp.name

        # Process completions
        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                status = "OK" if result['success'] else "FAILED"
                elapsed = result['elapsed']
                gpu = result['gpu']

                print(f"[{completed}/{n_experiments}] {exp_name} ({status}) - GPU {gpu} - {elapsed:.1f}s")

            except Exception as e:
                completed += 1
                print(f"[{completed}/{n_experiments}] {exp_name} (ERROR: {e})")
                results.append({'name': exp_name, 'success': False, 'error': str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel Noise-Resistant Training Campaign")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "4", "all"],
                        help="Which phase to run (1-4 or all)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--data", type=str,
                        default="data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5",
                        help="Path to HDF5 data file")
    parser.add_argument("--out", type=str, default="results/noise_resistant",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Just print experiments without running")

    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    # Select experiments based on phase
    if args.phase == "all":
        experiments = PHASE1_EXPERIMENTS + PHASE2_EXPERIMENTS + PHASE3_EXPERIMENTS + PHASE4_EXPERIMENTS
    else:
        phase_num = int(args.phase)
        experiments = ALL_PHASES[phase_num]

    print(f"Noise-Resistant Training Campaign")
    print(f"Phase: {args.phase}")
    print(f"Experiments: {len(experiments)}")
    print(f"GPUs: {gpu_ids}")
    print(f"Output: {args.out}")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN - Experiments to run:")
        for i, exp in enumerate(experiments):
            gpu = gpu_ids[i % len(gpu_ids)]
            print(f"  GPU {gpu}: {exp.name} ({exp.model}, {exp.loss}, seed={exp.seed})")
        return

    # Create output directory
    Path(args.out).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    results = run_parallel_campaign(
        experiments=experiments,
        gpu_ids=gpu_ids,
        data_path=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("CAMPAIGN SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/3600:.2f} hours")

    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r.get('success', False):
                print(f"  - {r['name']}: {r.get('error', 'unknown error')}")

    # Save results summary
    summary_file = Path(args.out) / "campaign_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'phase': args.phase,
            'total_experiments': len(results),
            'successful': successful,
            'failed': failed,
            'total_time_hours': total_time / 3600,
            'results': results
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
