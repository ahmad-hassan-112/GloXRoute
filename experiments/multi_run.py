import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")

SEEDS = [42, 43, 44, 45, 46]

# Start with normal scenario first.
# Later we will add failure, congestion, protocol, and scalability.
# SCENARIOS = [
#     {
#         "name": "normal",
#         "failure_rate": 0.0,
#         "congestion": False,
#         "protocol_mode": "soft_penalty",
#         "num_agents": 100,
#     },
# ]

SCENARIOS = [
    {
        "name": "normal",
        "failure_rate": 0.0,
        "congestion": False,
        "protocol_mode": "soft_penalty",
        "num_agents": 100,
    },
    {
        "name": "failure_10",
        "failure_rate": 0.10,
        "congestion": False,
        "protocol_mode": "soft_penalty",
        "num_agents": 100,
    },
    {
        "name": "congestion",
        "failure_rate": 0.0,
        "congestion": True,
        "protocol_mode": "soft_penalty",
        "num_agents": 100,
    },
    {
        "name": "protocol_hard",
        "failure_rate": 0.0,
        "congestion": False,
        "protocol_mode": "hard_block",
        "num_agents": 100,
    },
]


METRICS_TO_AGGREGATE = [
    "success_rate",
    "avg_latency",
    "avg_effective_latency",
    "p95_latency",
    "p95_effective_latency",
    "avg_hops",
    "avg_overhead",
    "explanation_rate",
    "avg_path_stretch",
    "loop_rate",
    "avg_load_on_path",
    "max_load_on_path",
    "protocol_mismatch_rate",
    "avg_decision_time_ms",
    "p95_decision_time_ms",
    "routing_score",
    "trust_score",
    "final_tradeoff_score",
]


def clean_for_new_run():
    shutil.rmtree("outputs/datasets", ignore_errors=True)
    shutil.rmtree("outputs/models/latest", ignore_errors=True)


def latest_run_dir(before_dirs):
    after_dirs = {
        p for p in OUTPUT_DIR.iterdir()
        if p.is_dir() and p.name != "datasets" and p.name != "models"
    }

    new_dirs = list(after_dirs - before_dirs)

    if not new_dirs:
        raise RuntimeError("No new output directory was created.")

    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def run_one_experiment(scenario, seed):
    print("=" * 80)
    print(f"Running scenario={scenario['name']} seed={seed}")
    print("=" * 80)

    before_dirs = {
        p for p in OUTPUT_DIR.iterdir()
        if p.is_dir()
    } if OUTPUT_DIR.exists() else set()

    clean_for_new_run()

    env = os.environ.copy()
    env["GLOX_SEED"] = str(seed)
    env["GLOX_SCENARIO_NAME"] = scenario["name"]
    env["GLOX_FAILURE_RATE"] = str(scenario["failure_rate"])
    env["GLOX_CONGESTION_SCENARIO"] = "true" if scenario["congestion"] else "false"
    env["GLOX_PROTOCOL_MODE"] = scenario["protocol_mode"]
    env["GLOX_NUM_AGENTS"] = str(scenario["num_agents"])

    subprocess.run(
        ["python3", "prepare_datasets.py"],
        check=True,
        env=env,
    )

    subprocess.run(
        ["python3", "main.py", "--mode", "all"],
        check=True,
        env=env,
    )

    run_dir = latest_run_dir(before_dirs)
    summary_path = run_dir / "comparison_summary_results.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    df["scenario"] = scenario["name"]
    df["seed"] = seed
    df["run_dir"] = str(run_dir)

    return df


def aggregate_results(raw_df):
    available_metrics = [
        metric for metric in METRICS_TO_AGGREGATE
        if metric in raw_df.columns
    ]

    summary = raw_df.groupby(["scenario", "method"])[available_metrics].agg(
        ["mean", "std"]
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    return summary


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = []

    for scenario in SCENARIOS:
        for seed in SEEDS:
            df = run_one_experiment(scenario, seed)
            all_results.append(df)

    raw_df = pd.concat(all_results, ignore_index=True)

    raw_path = OUTPUT_DIR / "multi_run_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = aggregate_results(raw_df)

    summary_path = OUTPUT_DIR / "multi_run_summary_mean_std.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nMulti-run complete.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Mean/std summary saved to: {summary_path}")

    print("\nMain model rows:")
    main_rows = summary_df[
        summary_df["method"].isin([
            "HybridXAI_HeuristicCoord",
            "HybridXAI_LearnedCoord",
            "ShortestPath",
            "Centralized",
            "DecentralizedGreedy",
            "BiRouterStyle",
        ])
    ]

    display_cols = [
        "scenario",
        "method",
        "avg_latency_mean",
        "avg_latency_std",
        "avg_effective_latency_mean",
        "avg_effective_latency_std",
        "routing_score_mean",
        "routing_score_std",
        "final_tradeoff_score_mean",
        "final_tradeoff_score_std",
    ]

    display_cols = [c for c in display_cols if c in main_rows.columns]

    print(main_rows[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()