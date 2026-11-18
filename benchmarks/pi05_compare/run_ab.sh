#!/bin/bash
# Timing rounds: each row runs once per round, in a fresh process, and the order rotates every round.
#
#   bash run_ab.sh GPU OUT_DIR ROUNDS ROW...
#
# GPU is the physical card number. Each run writes OUT_DIR/ROW_rN.json (timings), OUT_DIR/ROW_rN.npz (actions) and
# OUT_DIR/ROW_rN.peak (the card's memory before the run and its peak while the run lasted, from nvidia-smi).
#
# Rows:
#   jax, pt         openpi JAX and openpi PyTorch at openpi's own shape (3 camera slots, 200 prompt tokens)
#   jax585, pt585   the same at a bundle's shape (2 cameras, 73 prompt tokens)
#   bundle:NAME     the bundle in $BUNDLES/NAME
#   ref, ref_state  float32 references (openpi PyTorch): openpi's prompt, and LeRobot's prompt with the state
#   ref_state_3cam  the float32 reference for a bundle made with make_checkpoint.py --third-camera
#
# Settings, from the environment:
#   OPENPI_PY       python of openpi's environment            BUNDLE_PY   python of the lerobot-export environment
#   OPENPI_SRC      openpi checkout (for PYTHONPATH)          BUNDLES     folder of exported bundles
#   JAX_CKPT        openpi's pi05_libero JAX checkpoint       FRAMES      frames224.npz from make_frames.py
#   PT_CKPT         its PyTorch conversion
set -u
GPU=$1; OUT=$2; ROUNDS=$3; shift 3
ROWS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU BENCH_GPU=$GPU OMP_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false

run_one() {  # run_one NAME COMMAND...
  local name=$1; shift
  local base peak m pid
  base=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  peak=$base
  "$@" > "$OUT/$name.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ "${m:-0}" -gt "$peak" ] 2>/dev/null && peak=$m
    sleep 0.5
  done
  wait $pid
  local rc=$?
  echo "{\"baseline_mib\": $base, \"peak_mib\": $peak, \"rc\": $rc}" > "$OUT/$name.peak"
  echo "$(date '+%F %T') $name rc=$rc $(grep -h '^RESULT' "$OUT/$name.log" | grep -oE '"model_ms": \{"p50": [0-9.]+')"
}

openpi() {  # openpi NAME CKPT ARGS...
  local name=$1 ckpt=$2; shift 2
  PYTHONPATH="$OPENPI_SRC/src:$OPENPI_SRC/packages/openpi-client/src" run_one "$name" \
    "$OPENPI_PY" "$HERE/bench_openpi.py" --ckpt "$ckpt" --frames "$FRAMES" --out "$OUT/$name.json" \
    --actions "$OUT/$name.npz" "$@"
}

run_row() {  # run_row ROW ROUND
  local row=$1 name="${1#bundle:}_r$2"
  case $row in
    jax) openpi "$name" "$JAX_CKPT" ;;
    pt) openpi "$name" "$PT_CKPT" --warm 5 ;;
    jax585) openpi "$name" "$JAX_CKPT" --two-cams --max-token-len 73 ;;
    pt585) openpi "$name" "$PT_CKPT" --warm 5 --two-cams --max-token-len 73 ;;
    ref) openpi ref "$PT_CKPT" --f32 --chunks 3 --warm 1 ;;
    ref_state) openpi ref_state "$PT_CKPT" --f32 --discrete-state --chunks 3 --warm 1 ;;
    ref_state_3cam) openpi ref_state_3cam "$PT_CKPT" --f32 --discrete-state --unmask-third --chunks 3 --warm 1 ;;
    bundle:*)
      run_one "$name" "$BUNDLE_PY" "$HERE/bench_bundle.py" "$BUNDLES/${row#bundle:}" --frames "$FRAMES" \
        --out "$OUT/$name.json" --actions "$OUT/$name.npz" ;;
    *) echo "unknown row $row"; return 1 ;;
  esac
}

n=${#ROWS[@]}
for r in $(seq 1 "$ROUNDS"); do
  for i in $(seq 0 $((n - 1))); do
    run_row "${ROWS[$(( (i + r - 1) % n ))]}" "$r"
  done
done
