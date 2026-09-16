#!/usr/bin/env python3
"""
GloXRoute re-analysis: everything derivable from the per-task CSVs
without re-running experiments.

Produces:
  1. component_table.csv     - per-config, per-method means with 95% CIs
  2. weight_sensitivity.csv  - composite score under alternative weightings
  3. overhead_table.csv      - communication overhead + decision time
  4. override_diagnostics.csv- coordinator-override rate vs outcomes
  5. stats_table.csv         - Wilcoxon vs ShortestPath, Holm-adjusted, Cohen's d

Usage:
    python reanalysis.py --root results/ --out analysis/

Expects per-task CSVs discoverable under --root. Seed and configuration are
inferred from the path; adjust PATH_RE below if your layout differs.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Path layout:
#   outputs/<timestamp>/<topology_folder>/seed_<N>/all_results.csv
#
# topology_folder examples:
#   barabasi_albert            → topology=BA, condition=normal
#   barabasi_albert_congestion → topology=BA, condition=congestion
#   barabasi_albert_fail0.10   → topology=BA, condition=failure
#   watts_strogatz             → topology=WS, condition=normal
#   erdos_renyi_congestion     → topology=ER, condition=congestion
# ---------------------------------------------------------------------------
PATH_RE = re.compile(
    r"[/\\](?P<topo_folder>(?:barabasi_albert|watts_strogatz|erdos_renyi)[^/\\]*)"
    r"[/\\]seed[_-]?(?P<seed>\d+)[/\\]all_results\.csv$",
    re.IGNORECASE,
)

TOPO_MAP = {
    "barabasi_albert": "BA",
    "watts_strogatz":  "WS",
    "erdos_renyi":     "ER",
}

def parse_topo_condition(folder: str):
    """Return (topology_code, condition) from a folder name like
    'barabasi_albert_congestion' or 'erdos_renyi_fail0.10'."""
    folder = folder.lower()
    for key, code in TOPO_MAP.items():
        if folder.startswith(key):
            suffix = folder[len(key):]          # '', '_congestion', '_fail0.10'
            if "congestion" in suffix:
                condition = "congestion"
            elif "fail" in suffix:
                condition = "failure"
            else:
                condition = "normal"
            return code, condition
    return folder, "normal"   # fallback

PROPOSED = "HybridXAI_HeuristicCoord"
REFERENCE = "ShortestPath"

# Components that make up the composite routing score.
# Directions: +1 = higher is better, -1 = lower is better.
COMPONENTS = {
    "success": +1,
    "effective_latency": -1,
    "deadline_miss": -1,
    "protocol_mismatch_rate": -1,
    "path_stretch": -1,
}

# The weighting used in the paper, plus alternatives for the sensitivity check.
WEIGHT_SETS = {
    "paper": {
        "success": 0.30,
        "effective_latency": 0.30,
        "deadline_miss": 0.20,
        "protocol_mismatch_rate": 0.10,
        "path_stretch": 0.10,
    },
    "equal": {k: 0.20 for k in COMPONENTS},
    "latency_heavy": {
        "success": 0.20,
        "effective_latency": 0.50,
        "deadline_miss": 0.15,
        "protocol_mismatch_rate": 0.10,
        "path_stretch": 0.05,
    },
    "deadline_heavy": {
        "success": 0.20,
        "effective_latency": 0.20,
        "deadline_miss": 0.45,
        "protocol_mismatch_rate": 0.10,
        "path_stretch": 0.05,
    },
    "no_latency": {
        "success": 0.40,
        "effective_latency": 0.00,
        "deadline_miss": 0.30,
        "protocol_mismatch_rate": 0.15,
        "path_stretch": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all(root: Path) -> pd.DataFrame:
    # Collect all matching CSVs; if a (topo_folder, seed) appears in multiple
    # timestamp runs, keep only the latest timestamp to avoid double-counting.
    seen: dict[tuple, Path] = {}
    for csv_path in sorted(root.rglob("all_results.csv")):
        # Skip the legacy 'old claude' run
        if "old" in str(csv_path).lower():
            continue
        m = PATH_RE.search(str(csv_path))
        if not m:
            continue
        key = (m.group("topo_folder").lower(), m.group("seed"))
        # sorted() above means later timestamps overwrite earlier ones — good.
        seen[key] = (csv_path, m)

    frames = []
    for (topo_folder, seed_str), (csv_path, m) in seen.items():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  WARNING: could not read {csv_path}: {e}")
            continue
        if "method" not in df.columns:
            continue
        topo_code, condition = parse_topo_condition(m.group("topo_folder"))
        df["topology"]  = topo_code
        df["condition"] = condition
        df["seed"]      = int(m.group("seed"))
        df["config"]    = topo_code + "/" + condition
        frames.append(df)

    if not frames:
        raise SystemExit(
            f"No all_results.csv files matched under {root}.\n"
            f"Expected layout: <root>/<timestamp>/<topology_folder>/seed_<N>/all_results.csv\n"
            f"Check that --root points to the 'outputs/' directory."
        )

    data = pd.concat(frames, ignore_index=True)
    for col in ("success", "deadline_miss", "loop_detected", "explanation_available"):
        if col in data.columns:
            data[col] = data[col].astype(str).str.lower().map(
                {"true": 1.0, "false": 0.0}
            ).fillna(data[col])
    # Guard against infinite effective latency being treated as a success.
    data["effective_latency"] = pd.to_numeric(
        data["effective_latency"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    n_seeds = data.groupby("config")["seed"].nunique()
    print(f"Loaded {len(data):,} task rows")
    print(f"Configurations: {data['config'].nunique()}")
    print("Seeds per configuration:")
    print(n_seeds.to_string())
    return data


# ---------------------------------------------------------------------------
# 1. Component table
# ---------------------------------------------------------------------------
def component_table(data: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        data.groupby(["config", "method", "seed"])[list(COMPONENTS)]
        .mean()
        .reset_index()
    )
    rows = []
    for (config, method), grp in per_seed.groupby(["config", "method"]):
        row = {"config": config, "method": method, "n_seeds": len(grp)}
        for comp in COMPONENTS:
            vals = grp[comp].dropna().values
            if len(vals) == 0:
                continue
            mean = vals.mean()
            # 95% CI on the mean across seeds
            half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row[f"{comp}_mean"] = mean
            row[f"{comp}_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["config", "method"])


# ---------------------------------------------------------------------------
# 2. Weight sensitivity
# ---------------------------------------------------------------------------
def normalise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise each component within each (config, seed) so that
    higher always means better, then recombine under each weight set."""
    out = per_seed.copy()
    for comp, direction in COMPONENTS.items():
        def _norm(g):
            v = g[comp]
            lo, hi = v.min(), v.max()
            if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                return pd.Series(0.5, index=g.index)
            scaled = (v - lo) / (hi - lo)
            return scaled if direction > 0 else 1.0 - scaled

        out[f"n_{comp}"] = (
            per_seed.groupby(["config", "seed"], group_keys=False).apply(_norm)
        )
    return out


def weight_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        data.groupby(["config", "method", "seed"])[list(COMPONENTS)]
        .mean()
        .reset_index()
    )
    per_seed = normalise(per_seed)

    rows = []
    for name, weights in WEIGHT_SETS.items():
        score = sum(
            per_seed[f"n_{comp}"] * w for comp, w in weights.items() if w > 0
        )
        tmp = per_seed[["config", "method", "seed"]].copy()
        tmp["score"] = score
        agg = tmp.groupby(["config", "method"])["score"].mean().reset_index()
        agg["weighting"] = name
        # Rank within each configuration; 1 = best.
        agg["rank"] = agg.groupby("config")["score"].rank(ascending=False)
        rows.append(agg)

    out = pd.concat(rows, ignore_index=True)
    pivot = out.pivot_table(
        index=["config", "method"], columns="weighting", values=["score", "rank"]
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    return pivot.reset_index()


# ---------------------------------------------------------------------------
# 3. Overhead
# ---------------------------------------------------------------------------
def overhead_table(data: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("communication_overhead", "decision_time_ms", "hops") if c in data]
    per_seed = data.groupby(["config", "method", "seed"])[cols].mean().reset_index()
    agg = per_seed.groupby(["config", "method"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 4. Coordinator-override diagnostics
# ---------------------------------------------------------------------------
def override_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """For the proposed method, does overriding the coordinator's recommended
    next hop correlate with worse outcomes?

    Works with two explanation formats:
      A) explanation_available=True rows have a text explanation string
         containing 'recommended_by_global=0.0' or '=1.0' per hop.
      B) explanation column contains a JSON/dict-like string with a
         'recommended_by_global' key at the task level (0 or 1).
    """
    sub = data[data["method"] == PROPOSED].copy()

    # Only rows where GloXRoute actually produced an explanation.
    # Note: load_all converts bool columns to 1.0/0.0 floats.
    if "explanation_available" in sub.columns:
        sub = sub[sub["explanation_available"].astype(float) == 1.0]

    if sub.empty or "explanation" not in sub.columns:
        print("  NOTE: override_diagnostics skipped — no explanation_available=True rows "
              f"found for method '{PROPOSED}'. Check explanation_available column.")
        return pd.DataFrame()

    def override_frac(text):
        if not isinstance(text, str) or not text.strip():
            return np.nan
        # Format A: per-hop flags in the explanation string
        flags = re.findall(r"recommended_by_global['\"]?\s*[:=]\s*([01])(?:\.0)?", text)
        if flags:
            flags = [int(f) for f in flags]
            return 1.0 - (sum(flags) / len(flags))
        # Format B: single boolean at task level
        m = re.search(r"['\"]?recommended_by_global['\"]?\s*[:=]\s*(true|false|1|0)", text, re.I)
        if m:
            val = m.group(1).lower()
            followed = 1 if val in ("true", "1") else 0
            return 1.0 - followed
        return np.nan

    sub["override_frac"] = sub["explanation"].map(override_frac)

    n_parseable = sub["override_frac"].notna().sum()
    if n_parseable == 0:
        # Print a sample explanation so we can adjust the regex if needed
        sample = sub["explanation"].dropna().iloc[0] if not sub["explanation"].dropna().empty else ""
        print(f"  NOTE: override_diagnostics — could not parse override signal.\n"
              f"  Sample explanation (first 300 chars):\n  {str(sample)[:300]}")
        return pd.DataFrame()

    sub["any_override"] = sub["override_frac"].fillna(0) > 0

    rows = []
    for config, grp in sub.groupby("config"):
        g = grp.dropna(subset=["override_frac"])
        if g.empty:
            continue
        for label, mask in (("override", g["any_override"]), ("no_override", ~g["any_override"])):
            part = g[mask]
            if part.empty:
                continue
            rows.append(
                {
                    "config": config,
                    "group": label,
                    "n_tasks": len(part),
                    "deadline_miss_rate": part["deadline_miss"].mean(),
                    "mean_path_stretch": part["path_stretch"].mean(),
                    "mean_effective_latency": part["effective_latency"].mean(),
                    "mean_hops": part["hops"].mean(),
                }
            )
        rows.append(
            {
                "config": config,
                "group": "ALL",
                "n_tasks": len(g),
                "override_task_rate": g["any_override"].mean(),
                "mean_override_frac": g["override_frac"].mean(),
                "deadline_miss_rate": g["deadline_miss"].mean(),
                "mean_path_stretch": g["path_stretch"].mean(),
                "mean_effective_latency": g["effective_latency"].mean(),
                "mean_hops": g["hops"].mean(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Statistics: Wilcoxon + Holm + Cohen's d
# ---------------------------------------------------------------------------
def cohens_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def holm(pvals):
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def stats_table(data: pd.DataFrame, metric: str = "effective_latency") -> pd.DataFrame:
    per_seed = (
        data.groupby(["config", "method", "seed"])[metric].mean().reset_index()
    )
    records = []
    for config, grp in per_seed.groupby("config"):
        ref = grp[grp["method"] == REFERENCE].set_index("seed")[metric]
        for method, mg in grp.groupby("method"):
            if method == REFERENCE:
                continue
            m = mg.set_index("seed")[metric]
            common = ref.index.intersection(m.index)
            if len(common) < 3:
                continue
            a, b = m.loc[common].values, ref.loc[common].values
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, 1.0
            records.append(
                {
                    "config": config,
                    "method": method,
                    "metric": metric,
                    "n_seeds": len(common),
                    "mean_method": a.mean(),
                    "mean_reference": b.mean(),
                    "delta": a.mean() - b.mean(),
                    "wilcoxon_stat": stat,
                    "p_raw": p,
                    "cohens_d": cohens_d_paired(a, b),
                }
            )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_raw"].values)
    out["significant_holm_05"] = out["p_holm"] < 0.05
    return out.sort_values(["config", "method"])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="directory holding per-task CSVs")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_all(args.root)

    # Sanity checks worth knowing before anything else.
    print("\n--- sanity ---")
    if "loop_detected" in data:
        print(f"tasks with loops: {int(data['loop_detected'].sum())}")
    print(f"overall success rate: {data['success'].mean():.4f}")
    by_method = data.groupby("method")["success"].mean()
    print("success rate by method:")
    print(by_method.round(4).to_string())

    outputs = {
        "component_table.csv": component_table(data),
        "weight_sensitivity.csv": weight_sensitivity(data),
        "overhead_table.csv": overhead_table(data),
        "override_diagnostics.csv": override_diagnostics(data),
    }
    for metric in ("effective_latency", "deadline_miss", "protocol_mismatch_rate"):
        outputs[f"stats_{metric}.csv"] = stats_table(data, metric)

    for name, df in outputs.items():
        if df is None or df.empty:
            print(f"skipped {name} (no data)")
            continue
        df.to_csv(args.out / name, index=False)
        print(f"wrote {args.out / name}  ({len(df)} rows)")

    # Headline check: does GloXRoute's ranking survive reweighting?
    ws = outputs["weight_sensitivity.csv"]
    if not ws.empty:
        rank_cols = [c for c in ws.columns if c.startswith("rank_")]
        prop = ws[ws["method"] == PROPOSED]
        print("\n--- rank of proposed method under each weighting ---")
        print(prop[["config"] + rank_cols].to_string(index=False))


if __name__ == "__main__":
    main()