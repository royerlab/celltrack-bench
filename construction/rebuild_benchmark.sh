#!/usr/bin/env bash
# Build data/hard and data/full from the saved tracker outputs and the raw sequences, verify every
# file against data/CHECKSUMS.sha256 (the released build), and install the result into data/.
#
#   RAW_DATA=/path/to/raw_data construction/rebuild_benchmark.sh
#
# Environment:
#   RAW_DATA         directory containing datasets/<dataset>/<sequence>{,_GT/TRA} (CTC layout)
#   TRACKER_OUTPUTS  directory containing evaluation/... (default: tracker_outputs/ in this repository)
#   WORK             workspace to create (default: runs/rebuild); must not exist
#   PYTHON           interpreter with construction/requirements.txt and this package installed
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RAW_DATA="$(cd "${RAW_DATA:?set RAW_DATA to the directory containing datasets/}" && pwd)"
TRACKER_OUTPUTS="$(cd "${TRACKER_OUTPUTS:-$REPO/tracker_outputs}" && pwd)"
WORK="${WORK:-$REPO/runs/rebuild}"
PYTHON="${PYTHON:-python3}"
A=evaluation/errors_gt_linking/trackastra_hoct_hungarian_analysis

[[ -e "$WORK" ]] && { echo "workspace exists: $WORK" >&2; exit 1; }
for g in hoct_groupcv trackastra_node_only_groupcv; do
  n=$(find "$TRACKER_OUTPUTS/evaluation/$g/full_graphs" -maxdepth 3 -name '*.geff' | wc -l)
  [[ "$n" -eq 35 ]] || { echo "expected 35 $g graphs, found $n" >&2; exit 1; }
done
missing=0
while read -r ds seqs; do
  for s in $seqs; do
    for d in "$RAW_DATA/datasets/$ds/$s" "$RAW_DATA/datasets/$ds/${s}_GT/TRA"; do
      [[ -d "$d" ]] || { echo "missing raw data: $d" >&2; missing=1; }
    done
  done
done <<'EOF'
BF-C2DL-HSC 01 02
BF-C2DL-MuSC 01 02
DIC-C2DH-HeLa 01 02
Fluo-C2DL-Huh7 01 02
Fluo-C2DL-MSC 01 02
Fluo-N2DH-GOWT1 01 02
Fluo-N2DH-SIM+ 01 02
Fluo-N2DL-HeLa 01 02
PhC-C2DH-U373 01 02
PhC-C2DL-PSC 01 02
Bacteria-cib 01 02
Bacteria-metA 01 02 03 04 05
Bacteria-pheA 05
Bacteria-rpsM 01 02 07
Bacteria-trpL 03 04 05 06
EOF
[[ "$missing" -eq 0 ]] || exit 1
"$PYTHON" -c "import celltrack_bench, tracksdata, zarr, scipy" \
  || { echo "$PYTHON lacks celltrack_bench or construction/requirements.txt" >&2; exit 1; }

# The scripts resolve every path relative to the workspace root and record some of them in the items,
# so the tracker outputs are copied in (a symlink would resolve outside the workspace).
mkdir -p "$WORK"
cp -r "$HERE/scripts" "$WORK/scripts"
cp -r "$TRACKER_OUTPUTS/evaluation" "$WORK/evaluation"
ln -s "$RAW_DATA/datasets" "$WORK/datasets"
cd "$WORK"

echo "== 1/4 three-tracker error analysis"
"$PYTHON" scripts/analyze_three_tracker_errors.py > analyze.log
echo "== 2/4 hard-mining selection (top 64 and top 256 per dataset and split)"
"$PYTHON" scripts/select_three_tracker_error_pool_by_split.py > select64.log
"$PYTHON" scripts/select_three_tracker_error_pool_by_split.py --per-dataset-per-split 256 \
  --output-dir "$A/selected_top256_per_split_per_dataset" > select256.log
echo "== 3/4 item builds"
for pair in 64:celltrack_bench_consensus_v2 256:celltrack_bench_consensus_v2_xl; do
  k=${pair%%:*}; build=${pair##*:}
  for split in validation test; do
    echo "   $build/$split"
    "$PYTHON" scripts/build_hoct_graph_benchmarks.py \
      --selected-anchors "$A/selected_top${k}_per_split_per_dataset/tracker_count_then_hardness.csv" \
      --data-split "$split" --output-dir "evaluation/$build/$split" --skip-render --overwrite > "build_${build}_${split}.log"
    "$PYTHON" scripts/validate_selected_hoct_graph_benchmarks.py --root "evaluation/$build/$split" \
      --data-split "$split" --max-selection-rank "$k" > "validate_${build}_${split}.log"
  done
done
echo "== 4/4 export, verify against the released checksums, install into data/"
"$PYTHON" "$HERE/export_release.py" --workspace "$WORK" --build celltrack_bench_consensus_v2 \
  --benchmark-id celltrack-bench-consensus-v2 --output "$WORK/release/hard"
"$PYTHON" "$HERE/export_release.py" --workspace "$WORK" --build celltrack_bench_consensus_v2_xl \
  --benchmark-id celltrack-bench-consensus-v2-xl --output "$WORK/release/full"

(cd "$WORK/release" && sha256sum --quiet -c "$REPO/data/CHECKSUMS.sha256") \
  || { echo "rebuilt files do NOT match data/CHECKSUMS.sha256 (see above); data/ left untouched" >&2; exit 1; }
expected=$(wc -l < "$REPO/data/CHECKSUMS.sha256")
produced=$(cd "$WORK/release" && find hard full -type f | wc -l)
[[ "$produced" -eq "$expected" ]] || { echo "rebuild produced $produced files, expected $expected" >&2; exit 1; }
echo "all $expected rebuilt files match the released checksums"

for bench in hard full; do
  if [[ -e "$REPO/data/$bench" ]]; then
    diff -r -q "$WORK/release/$bench" "$REPO/data/$bench" >/dev/null \
      && echo "data/$bench already present and identical" \
      || { echo "data/$bench exists and differs; remove it and re-run to install" >&2; exit 1; }
  else
    cp -r "$WORK/release/$bench" "$REPO/data/$bench"
    echo "installed data/$bench"
  fi
done
