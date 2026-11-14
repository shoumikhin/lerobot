#!/bin/bash
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# usage: run_ab.sh <gpu index> <bundle dir> <seq.npz> <out dir> [passes=3]
#
# Alternating passes on one GPU. Each pass is a fresh process per side that loads, runs WARMUP + COUNT chunks of
# the recorded sequence, and exits. Then summarize.py prints each pass's median and the median of the medians.
#
# Environment:
#   BUNDLE_PY       python of the LeRobot environment (bundle and lerobot sides)    default: python
#   RUNWAY_PY       python of Runway's environment (runway sides)                    default: python
#   RUNWAY_REPO     a checkout of github.com/runwayml/praxis-1-oss                   needed for the runway sides
#   RUNWAY_WEIGHTS  folder with Runway's checkpoints/ and weights/                   default: RUNWAY_REPO
#   RUNWAY_CKPT     checkpoint folder, relative to RUNWAY_WEIGHTS                    default: checkpoints/praxis-1-droid
#   LEROBOT_SRC     LeRobot sources to use                                           default: this repository's src
#   SIDES           any of: bundle runway runway_cr lerobot                          default: bundle runway runway_cr
#   WARMUP, COUNT   warm-up and timed chunks per pass                                default: 3 and 30
#
# runway_cr is Runway with the client-side resize (see runway_side.py); lerobot is LeRobot's eager policy.
# Before each pass the script waits until no other process runs kernels on the GPU (up to 10 minutes). A pass
# during which another process ran on the GPU is thrown away and run again (up to 2 more times). The script exits
# non-zero when a pass has no clean run, or when summarize.py finds a side short of passes or the sides' sampler
# settings differ. The output directory must be new or empty, so that only this run's logs are summarized.
set -u
G=$1; B=$2; SEQ=$3; D=$4; N=${5:-3}
if [ -n "$(ls -A "$D" 2>/dev/null)" ]; then echo "$D is not empty: give each run a new output directory"; exit 2; fi
read -r -d '' -a sides <<< "${SIDES:-bundle runway runway_cr}" || true
[ ${#sides[@]} -gt 0 ] || { echo "SIDES names no side"; exit 2; }
for side in "${sides[@]}"; do
  case $side in bundle|runway|runway_cr|lerobot) ;; *) echo "unknown side '$side' in SIDES"; exit 2 ;; esac
done
H=$(cd "$(dirname "$0")" && pwd)
SRC=${LEROBOT_SRC:-$H/../../src}
BUNDLE_PY=${BUNDLE_PY:-python}; RUNWAY_PY=${RUNWAY_PY:-python}
RUNWAY_REPO=${RUNWAY_REPO:-}; RUNWAY_WEIGHTS=${RUNWAY_WEIGHTS:-$RUNWAY_REPO}
RUNWAY_CKPT=${RUNWAY_CKPT:-checkpoints/praxis-1-droid}
W=${WARMUP:-3}; K=${COUNT:-30}; CH=$((W + K))
mkdir -p "$D"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$G

busy_pids() {  # processes running kernels on the GPU now (an idle context does not count)
  nvidia-smi pmon -c 1 -s u -i "$G" 2>/dev/null | awk '$1 !~ /^#/ && $4 != "-" && $4+0 > 0 {print $2}'
}
watch_others() {  # log every busy process outside the pass's own process group, once a second
  local root=$1 out=$2
  (
    while kill -0 "$root" 2>/dev/null; do
      own=" $(pgrep -g "$(ps -o pgid= "$root" | tr -d ' ')" 2>/dev/null | tr '\n' ' ') "
      for q in $(busy_pids); do case "$own" in *" $q "*) ;; *) echo "$q" >> "$out" ;; esac; done
      sleep 1
    done
  ) &
}

common=(--warmup "$W" --count "$K" --chunks "$CH")
runway=(--weights-dir "$RUNWAY_WEIGHTS" --checkpoint "$RUNWAY_CKPT" --rotation "$SRC/lerobot/utils/rotation.py")
failed=0
for i in $(seq 1 "$N"); do for side in "${sides[@]}"; do clean=0; for attempt in 1 2 3; do
  for _ in $(seq 1 40); do [ -z "$(busy_pids)" ] && break; sleep 15; done
  O=$D/${side}_pass$i; rm -f "$O.others"
  case $side in
    bundle) cmd=(env PYTHONPATH="$SRC" "$BUNDLE_PY" "$H/bundle_side.py" "$B" "$SEQ" "$O.npz") ;;
    lerobot) cmd=(env PYTHONPATH="$SRC" "$BUNDLE_PY" "$H/lerobot_eager.py" "$SEQ" "$O.npz") ;;
    runway) cmd=("$RUNWAY_PY" "$H/runway_side.py" "$RUNWAY_REPO" "$SEQ" "$O.npz" "${runway[@]}") ;;
    runway_cr) cmd=("$RUNWAY_PY" "$H/runway_side.py" "$RUNWAY_REPO" "$SEQ" "$O.npz" "${runway[@]}" --client-resize) ;;
  esac
  setsid "${cmd[@]}" "${common[@]}" > "$O.log" 2>&1 &
  P=$!; watch_others $P "$O.others"; WP=$!; wait $P; rc=$?; wait $WP
  echo "PASS $i $side rc=$rc $(grep '^SUMMARY' "$O.log" | cut -c9-)"
  if [ $rc != 0 ]; then
    tail -5 "$O.log"; mv "$O.log" "$O.failed$attempt.log"; continue
  fi
  if [ ! -s "$O.others" ]; then clean=1; break; fi
  echo "PASS $i $side: other processes ran on the GPU ($(sort -u "$O.others" | tr '\n' ' ')), running it again"
  mv "$O.log" "$O.busy$attempt.log"
done
  [ $clean = 1 ] || { echo "PASS $i $side: no clean run in 3 attempts"; failed=1; }
done; done
"$BUNDLE_PY" "$H/summarize.py" "$D" "$N" "${sides[@]}" || failed=1
exit $failed
