#!/usr/bin/env bash
# gather_paper_csvs.sh
# Copies only the small, paper-relevant CSVs into paper_data/
# Nothing large (no per-seed all_results.csv, no raw files)
#
# Usage:
#   bash gather_paper_csvs.sh outputs/2026-09-03_15-46-09
#   bash gather_paper_csvs.sh outputs/2026-09-03_15-46-09 paper_data

set -euo pipefail

RUN="${1:?Usage: bash gather_paper_csvs.sh <run_folder> [dest]}"
DEST="${2:-paper_data}"

if [[ ! -d "$RUN" ]]; then
    echo "ERROR: run folder not found: $RUN"; exit 1
fi

mkdir -p "$DEST"
echo "Source : $RUN"
echo "Dest   : $DEST"
echo

ok=0; miss=0

cp_file() {
    local src="$1" dest_name="$2"
    if [[ -f "$src" ]]; then
        cp "$src" "$DEST/$dest_name"
        local size; size=$(du -h "$src" | cut -f1)
        printf "  [OK]  %-50s  %s\n" "$dest_name" "$size"
        (( ok++ )) || true
    else
        printf "  [!!]  %-50s  NOT FOUND: %s\n" "$dest_name" "$src"
        (( miss++ )) || true
    fi
}

echo "── MAIN MATRIX (multiseed_all.csv per topology) ──────────────────────"
cp_file "$RUN/barabasi_albert/multiseed_all.csv"            "ba_normal_multiseed_all.csv"
cp_file "$RUN/barabasi_albert_congestion/multiseed_all.csv" "ba_congestion_multiseed_all.csv"
cp_file "$RUN/barabasi_albert_fail0.10/multiseed_all.csv"   "ba_failure_multiseed_all.csv"
cp_file "$RUN/watts_strogatz/multiseed_all.csv"             "ws_normal_multiseed_all.csv"
cp_file "$RUN/watts_strogatz_congestion/multiseed_all.csv"  "ws_congestion_multiseed_all.csv"
cp_file "$RUN/watts_strogatz_fail0.10/multiseed_all.csv"    "ws_failure_multiseed_all.csv"
cp_file "$RUN/erdos_renyi/multiseed_all.csv"                "er_normal_multiseed_all.csv"
cp_file "$RUN/erdos_renyi_congestion/multiseed_all.csv"     "er_congestion_multiseed_all.csv"
cp_file "$RUN/erdos_renyi_fail0.10/multiseed_all.csv"       "er_failure_multiseed_all.csv"

echo
echo "── ABLATION ──────────────────────────────────────────────────────────"
cp_file "$RUN/ablation/ablation_all.csv"    "ablation_all.csv"
cp_file "$RUN/ablation/ablation_mean.csv"   "ablation_mean.csv"
cp_file "$RUN/ablation/ablation_std.csv"    "ablation_std.csv"


echo
echo "── SCALABILITY ───────────────────────────────────────────────────────"
cp_file "$RUN/scalability/scalability_summary.csv"  "scalability_summary.csv"
cp_file "$RUN/scalability/scalability_summary1.csv" "scalability_summary1.csv"

echo
echo "── XAI ───────────────────────────────────────────────────────────────"
cp_file "$RUN/xai_eval/xai_summary.csv"          "xai_summary.csv"
cp_file "$RUN/xai_eval/feature_consistency.csv"  "feature_consistency.csv"
cp_file "$RUN/xai_eval/counterfactuals.csv"      "counterfactuals.csv"

echo
echo "── RUN LOG ───────────────────────────────────────────────────────────"
cp_file "$RUN/run.log" "run.log"

echo
echo "══════════════════════════════════════════════════════════════════════"
printf "  Done: %d copied, %d missing\n" "$ok" "$miss"
if (( miss == 0 )); then
    echo "  All files gathered. Upload the $DEST/ folder."
else
    echo "  Fix missing files above, then re-run."
fi
echo "  Total size: $(du -sh "$DEST" | cut -f1)"
echo "══════════════════════════════════════════════════════════════════════"