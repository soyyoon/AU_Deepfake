#!/usr/bin/env bash
# Parallel per-file streaming re-preprocessing: FaceForensics++ c23 + Celeb-DF v2.
# No big zip: each video is fetched on its own (stdlib urllib), run through Py-Feat
# (AUs+pose+emotions = 30ch, ~seq_len frames via auto-skip), saved as tiny .npy,
# then deleted. Py-Feat's AU stage is CPU-bound, so we run WORKERS shards in
# parallel across cores. Resumable: finished videos are skipped on re-run.
#
#   bash scripts/run_ffpp_celebdf.sh              # full run, default workers
#   WORKERS=14 bash scripts/run_ffpp_celebdf.sh   # more parallelism (watch GPU mem ~2.8GB/worker)
#   SMOKE=24 bash scripts/run_ffpp_celebdf.sh     # process only first 24 rows/dataset
#   ONLY=FFpp bash scripts/run_ffpp_celebdf.sh    # one dataset only (FFpp|CelebDF)
set -euo pipefail

cd "$(dirname "$0")/.."
PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
export WORKDIR=${WORKDIR:-./.cev_work}
mkdir -p "$WORKDIR" data outputs/logs
NCORES=$(nproc)
# Py-Feat's AU stage is effectively single-core per worker, so throughput scales
# with WORKERS up to NCORES. Spread workers round-robin over all GPUs (~1.8GB each).
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}; [ "$NGPUS" -lt 1 ] && NGPUS=1
WORKERS=${WORKERS:-$(( NGPUS * 8 ))}                      # 8/GPU (~15GB/GPU) = least-contention sweet spot
STAGGER=${STAGGER:-2}                                     # secs between worker starts (ease GPU cold-start)
SUBSET=${SUBSET:-1}                                       # 1 = balanced ~class-balanced subset (default), 0 = full
# Divide cores among workers so total threads ~= cores (no oversubscription thrash,
# no idle cores). Py-Feat's AU stage is CPU-bound and multithreaded.
OMP=${OMP_NUM_THREADS:-$(( NCORES / WORKERS ))}; [ "$OMP" -lt 1 ] && OMP=1
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP
echo "[cfg] WORKERS=$WORKERS OMP=$OMP cores=$NCORES gpus=$NGPUS stagger=${STAGGER}s"
LIMIT=""; [ "${SMOKE:-0}" != "0" ] && LIMIT="--limit ${SMOKE}"
ONLY=${ONLY:-}

run_one () {                                              # $1=kaggle slug  $2=dataset tag
  local slug="$1" tag="$2"
  [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && return 0
  echo "==================== $tag (WORKERS=$WORKERS) ===================="
  local fl="data/_filelist_${tag}.csv"
  if [ ! -s "$fl" ]; then                                # phase 1: list+label (no PYTHONNOUSERSITE)
    $PY scripts/build_filelist.py --slug "$slug" --tag "$tag"
  else
    echo "[filelist] reusing $fl ($(( $(wc -l < "$fl") - 1 )) videos)"
  fi
  if [ "$SUBSET" = "1" ]; then                           # class/method-balanced subset
    $PY scripts/make_subset.py --tag "$tag"
    fl="data/_filelist_${tag}_sub.csv"
  fi
  # phase 2: WORKERS parallel shards (PYTHONNOUSERSITE=1, urllib-only download)
  local pids=()
  for s in $(seq 0 $((WORKERS - 1))); do
    CUDA_VISIBLE_DEVICES=$(( s % NGPUS )) PYTHONNOUSERSITE=1 $PY scripts/stream_extract.py \
        --slug "$slug" --tag "$tag" --filelist "$fl" \
        --num-shards "$WORKERS" --shard "$s" $LIMIT \
        > "outputs/logs/${tag}_shard${s}.log" 2>&1 &
    pids+=($!)
    sleep "$STAGGER"                                      # stagger GPU cold-starts
  done
  echo "[run] $tag: launched ${#pids[@]} shard workers, waiting ..."
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[run] $tag shards finished (rc=$rc)"
  tail -n1 outputs/logs/${tag}_shard*.log
}

df -h . | tail -1
run_one "xdxd003/ff-c23"         "FFpp"
run_one "reubensuju/celeb-df-v2" "CelebDF"

echo "==================== FINALIZE ===================="
PYTHONNOUSERSITE=1 $PY scripts/finalize_metadata.py
df -h . | tail -1
echo "done."
