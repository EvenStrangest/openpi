#!/usr/bin/env bash
# pi0.5-LIBERO LIBERO-Plus perturbation eval on DGXA100S03 (Phase 2).
#
# Evaluates the pi05_libero checkpoint (the SAME server/checkpoint as the Phase 1
# base-gate -- server env untouched) on the LIBERO-Plus GEOMETRIC axes:
#   Camera Viewpoints / Robot Initial States / Objects Layout, + Language
#   Instructions as a control. 1 trial per perturbed task; agentview replay video
#   per task; per-task JSONL results (resumable).
#
# The sim client runs the committed openpi fork's examples/libero/main_libero_plus.py
# inside a `libero_plus` image (built here from Dockerfile.libero_plus). LIBERO-Plus
# itself (source + 9.5 GB assets) is BIND-MOUNTED from the pinned forks/LIBERO-plus
# submodule at /libero_plus and made the active `libero` package via PYTHONPATH +
# a LIBERO_CONFIG_PATH that repoints bddl_files/init_states/assets at it. The stock
# openpi_server image and its JAX venv are not touched.
#
# GPU plan (UUID-pinned; S03 usable A100-40GB are indices 0,1,3; index 2 = DGX Display).
# One persistent openpi_server per GPU + one sim client per GPU, task list round-
# robin-sharded 3 ways for balanced category mix and walltime.
#
# Usage:
#   SUITE=libero_spatial \
#   one-off-tools/... or examples/libero/run_libero_plus_s03.sh
# Env knobs (all optional; defaults shown):
#   SUITE=libero_spatial
#   CATEGORIES="Camera Viewpoints,Robot Initial States,Objects Layout,Language Instructions"
#   OPENPI_DIR=$HOME/terraforge-vla/forks/openpi
#   LIBERO_PLUS_DIR=$HOME/terraforge-vla/forks/LIBERO-plus
#   TASKLIST_DIR=$HOME/terraforge-vla/one-off-tools/libero_plus_tasklists
#   CACHE=$HOME/.cache/openpi
#   OUT=$HOME/terraforge-vla-results/openpi-pi05-libero-plus-<suite>
#   NSHARDS=3
#   SMOKE=0            # 1 == 1 task per category, single GPU, quick end-to-end check
set -uo pipefail

SUITE="${SUITE:-libero_spatial}"
CATEGORIES="${CATEGORIES:-Camera Viewpoints,Robot Initial States,Objects Layout,Language Instructions}"
OPENPI_DIR="${OPENPI_DIR:-$HOME/terraforge-vla/forks/openpi}"
LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-$HOME/terraforge-vla/forks/LIBERO-plus}"
TASKLIST_DIR="${TASKLIST_DIR:-$HOME/terraforge-vla/one-off-tools/libero_plus_tasklists}"
CACHE="${CACHE:-$HOME/.cache/openpi}"
OUT="${OUT:-$HOME/terraforge-vla-results/openpi-pi05-libero-plus-${SUITE#libero_}}"
NSHARDS="${NSHARDS:-3}"
SMOKE="${SMOKE:-0}"

GPU0=GPU-d44a862a-7089-ccc5-908d-36fff2005a9d
GPU1=GPU-91d023b4-91cb-7f79-c743-7f9b900eb139
GPU3=GPU-a7f06e5c-94e1-2efb-2c27-5c64670e6c0b
GPUS=("$GPU0" "$GPU1" "$GPU3")
PORTS=(8000 8001 8002)

mkdir -p "$OUT/logs" "$OUT/videos" "$OUT/results" "$OUT/shards"

# ------------------------------------------------------------------ image build
build_image () {
  if docker image inspect libero_plus >/dev/null 2>&1; then
    echo "[build] libero_plus image already present; skipping"
    return 0
  fi
  echo "[build] building libero_plus image $(date -Is)"
  ( cd "$OPENPI_DIR" && DOCKER_BUILDKIT=1 \
      docker build -t libero_plus -f examples/libero/Dockerfile.libero_plus . ) \
      2>&1 | tee "$OUT/logs/build.log"
}

# ------------------------------------------------------------------ task shards
make_shards () {
  local full="$TASKLIST_DIR/${SUITE}_full.tsv"
  [ -s "$full" ] || { echo "[shard] MISSING task list: $full"; exit 1; }
  # Filter to the requested categories (tab-delimited col 1).
  local filt="$OUT/shards/${SUITE}_filtered.tsv"
  awk -F'\t' -v cats="$CATEGORIES" '
    BEGIN { n=split(cats, a, ","); for (i=1;i<=n;i++){ gsub(/^ +| +$/,"",a[i]); keep[a[i]]=1 } }
    ($1 in keep) { print }
  ' "$full" > "$filt"
  local total; total=$(wc -l < "$filt")
  echo "[shard] $filt : $total tasks across categories [$CATEGORIES]"

  if [ "$SMOKE" = "1" ]; then
    # 1 task per category -> a single smoke shard.
    awk -F'\t' '!seen[$1]++' "$filt" > "$OUT/shards/${SUITE}_shard0.tsv"
    echo "[shard] SMOKE: $(wc -l < "$OUT/shards/${SUITE}_shard0.tsv") tasks (1/category) -> shard0"
    return 0
  fi

  # Round-robin split into NSHARDS (balances category mix + walltime).
  awk -F'\t' -v n="$NSHARDS" -v out="$OUT/shards/${SUITE}_shard" '
    { print >> (out (NR % n) ".tsv") }
  ' "$filt"
  for k in $(seq 0 $((NSHARDS-1))); do
    echo "[shard] shard$k : $(wc -l < "$OUT/shards/${SUITE}_shard${k}.tsv") tasks"
  done
}

# ------------------------------------------------------------------ policy server
start_server () {  # name uuid port
  local name="$1" uuid="$2" port="$3"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --network host \
    -e NVIDIA_VISIBLE_DEVICES="$uuid" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e OPENPI_DATA_HOME=/openpi_assets -e IS_DOCKER=true \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    -v "$OPENPI_DIR":/app \
    -v "$CACHE":/openpi_assets \
    openpi_server \
    /bin/bash -c "uv run scripts/serve_policy.py --env LIBERO --port $port" \
    >/dev/null
  echo "[server] $name uuid=$uuid port=$port"
}

wait_port () {  # port timeout_s
  local port="$1" to="${2:-3600}" i=0
  while ! (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; do
    sleep 5; i=$((i+5))
    if [ "$i" -ge "$to" ]; then echo "[wait_port] TIMEOUT port=$port after ${to}s"; return 1; fi
  done
  exec 3>&- 2>/dev/null || true
  echo "[wait_port] port=$port OPEN after ${i}s"
}

# ------------------------------------------------------------------ sim client
run_client () {  # shard_idx uuid port
  local k="$1" uuid="$2" port="$3"
  local shard="$OUT/shards/${SUITE}_shard${k}.tsv"
  [ -s "$shard" ] || { echo "[client] shard$k empty; skip"; return 0; }
  echo "[client] START shard=$k uuid=$uuid port=$port $(date -Is)" | tee -a "$OUT/logs/eval_shard${k}.log"
  docker run --rm --name "openpi_lp_cli_${SUITE}_${k}" --network host \
    -e NVIDIA_VISIBLE_DEVICES="$uuid" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e MUJOCO_GL=egl -e MUJOCO_EGL_DEVICE_ID=0 -e PYOPENGL_PLATFORM=egl \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/app:/app/packages/openpi-client/src:/libero_plus \
    -v "$OPENPI_DIR":/app \
    -v "$LIBERO_PLUS_DIR":/libero_plus:ro \
    -v "$OUT":/out \
    libero_plus \
    /bin/bash -c '
      set -e
      mkdir -p /tmp/lpcfg
      cat > /tmp/lpcfg/config.yaml <<EOF
benchmark_root: /libero_plus/libero/libero
bddl_files: /libero_plus/libero/libero/bddl_files
init_states: /libero_plus/libero/libero/init_files
datasets: /libero_plus/libero/datasets
assets: /libero_plus/libero/libero/assets
EOF
      export LIBERO_CONFIG_PATH=/tmp/lpcfg
      source /.venv/bin/activate
      python examples/libero/main_libero_plus.py \
        --args.task-suite-name '"$SUITE"' \
        --args.host 127.0.0.1 --args.port '"$port"' \
        --args.task-list /out/shards/'"${SUITE}_shard${k}.tsv"' \
        --args.results-jsonl /out/results/shard'"${k}"'.jsonl \
        --args.video-out-path /out/videos
    ' >> "$OUT/logs/eval_shard${k}.log" 2>&1
  echo "[client] DONE shard=$k $(date -Is)" | tee -a "$OUT/logs/eval_shard${k}.log"
}

# ------------------------------------------------------------------ main
main () {
  echo "=== pi0.5 LIBERO-Plus eval start suite=$SUITE smoke=$SMOKE $(date -Is) ==="
  build_image
  make_shards

  if [ "$SMOKE" = "1" ]; then
    start_server openpi_lp_srv_gpu0 "$GPU0" 8000
    wait_port 8000 3600 || { docker logs --tail 60 openpi_lp_srv_gpu0; exit 1; }
    run_client 0 "$GPU0" 8000
    docker rm -f openpi_lp_srv_gpu0 >/dev/null 2>&1 || true
    echo "=== SMOKE finished $(date -Is) ==="
    return 0
  fi

  # Warm the checkpoint cache with GPU0 first to avoid a 3-way load race.
  start_server openpi_lp_srv_gpu0 "$GPU0" 8000
  wait_port 8000 3600 || { docker logs --tail 60 openpi_lp_srv_gpu0; exit 1; }
  start_server openpi_lp_srv_gpu1 "$GPU1" 8001
  start_server openpi_lp_srv_gpu3 "$GPU3" 8002
  wait_port 8001 3600 || { docker logs --tail 60 openpi_lp_srv_gpu1; exit 1; }
  wait_port 8002 3600 || { docker logs --tail 60 openpi_lp_srv_gpu3; exit 1; }
  echo "=== all servers ready $(date -Is) ==="

  ( run_client 0 "$GPU0" 8000 ) & P0=$!
  ( run_client 1 "$GPU1" 8001 ) & P1=$!
  ( run_client 2 "$GPU3" 8002 ) & P2=$!
  wait $P0 $P1 $P2
  echo "=== all clients done $(date -Is) ==="

  docker rm -f openpi_lp_srv_gpu0 openpi_lp_srv_gpu1 openpi_lp_srv_gpu3 >/dev/null 2>&1 || true
  echo "=== pi0.5 LIBERO-Plus eval finished suite=$SUITE $(date -Is) ==="
}

main 2>&1 | tee -a "$OUT/logs/driver.log"
