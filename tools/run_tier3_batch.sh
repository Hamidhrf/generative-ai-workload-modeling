#!/bin/bash
# Unattended orchestration of 50 Tier 3 experiments (5 workloads x r=1..10).
#
# Invocation:
#   tmux new -s tier3-batch
#   nohup bash tools/run_tier3_batch.sh > /tmp/tier3_batch.log 2>&1 &
#   # detach with Ctrl-B D
set -uo pipefail

WORKLOADS=(bert yolo resnet152 gpt2 whisper)
REPLICAS=(1 2 3 4 5 6 7 8 9 10)
PROM_URL="${PROM_URL:-http://172.22.174.66:30090}"
DATA_DIR="${DATA_DIR:-data/raw/extension_tier3}"
STATE_FILE=".tier3_state.json"
JOURNAL="EXTENSION_JOURNAL.md"
POD_READY_TIMEOUT_DEFAULT=180
POD_READY_TIMEOUT_WHISPER=300
CLEANUP_TIMEOUT=60
EXPECTED_ROWS_MIN=700
EXPECTED_ROWS_MAX=720
EXPECTED_FILE_COUNT=22
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/tracegen/bin/python}"

# Per-pod non-GPU metrics: query is "sum by (pod)" (or raw per-pod), so row
# count scales with replica count, not just tick count.
PER_POD_METRICS=(pod_cpu_usage pod_memory_bytes pod_psi_cpu pod_psi_io pod_psi_memory pod_latency_avg pod_throughput)

CURRENT_WORKLOAD=""
CURRENT_R=""

halt() {
  local reason="$1"
  echo "BATCH HALTED AT ${CURRENT_WORKLOAD} r=${CURRENT_R}: ${reason}"
  if [ -n "$CURRENT_WORKLOAD" ]; then
    mark_status "$CURRENT_WORKLOAD" "$CURRENT_R" "failed" "$reason"
  fi
  exit 1
}

# ---------------------------------------------------------------------------
# State file helpers (python3 json — no jq dependency assumed)
# ---------------------------------------------------------------------------

init_state() {
  if [ -f "$STATE_FILE" ]; then
    return
  fi
  WORKLOADS_CSV=$(IFS=,; echo "${WORKLOADS[*]}")
  REPLICAS_CSV=$(IFS=,; echo "${REPLICAS[*]}")
  python3 - "$STATE_FILE" "$WORKLOADS_CSV" "$REPLICAS_CSV" <<'PYEOF'
import json, sys
state_file, workloads_csv, replicas_csv = sys.argv[1:4]
workloads = workloads_csv.split(",")
replicas = [int(x) for x in replicas_csv.split(",")]
entries = []
idx = 0
for w in workloads:
    for r in replicas:
        entries.append({
            "index": idx, "workload": w, "replicas": r, "status": "pending",
            "csv_paths": [], "started_at": None, "finished_at": None, "error_message": None
        })
        idx += 1
with open(state_file, "w") as f:
    json.dump(entries, f, indent=2)
print(f"Initialized {state_file} with {len(entries)} entries")
PYEOF
}

check_in_progress_crash() {
  python3 - "$STATE_FILE" <<'PYEOF'
import json, sys
state = json.load(open(sys.argv[1]))
crashed = [e for e in state if e["status"] == "in_progress"]
for e in crashed:
    print(f"{e['workload']} r={e['replicas']}")
sys.exit(1 if crashed else 0)
PYEOF
}

mark_status() {
  local workload="$1" replicas="$2" status="$3" error_msg="${4:-}"
  python3 - "$STATE_FILE" "$workload" "$replicas" "$status" "$error_msg" <<'PYEOF'
import json, sys
from datetime import datetime, timezone
state_file, workload, replicas, status, error_msg = sys.argv[1:6]
replicas = int(replicas)
state = json.load(open(state_file))
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
for e in state:
    if e["workload"] == workload and e["replicas"] == replicas:
        e["status"] = status
        if status == "in_progress":
            e["started_at"] = now
        if status in ("success", "failed"):
            e["finished_at"] = now
        if error_msg:
            e["error_message"] = error_msg
        break
with open(state_file, "w") as f:
    json.dump(state, f, indent=2)
PYEOF
}

set_csv_paths() {
  local workload="$1" replicas="$2" dir="$3"
  python3 - "$STATE_FILE" "$workload" "$replicas" "$dir" <<'PYEOF'
import json, sys, os
state_file, workload, replicas, d = sys.argv[1:5]
replicas = int(replicas)
state = json.load(open(state_file))
paths = sorted(os.path.join(d, f) for f in os.listdir(d))
for e in state:
    if e["workload"] == workload and e["replicas"] == replicas:
        e["csv_paths"] = paths
        break
with open(state_file, "w") as f:
    json.dump(state, f, indent=2)
PYEOF
}

find_start_index() {
  python3 - "$STATE_FILE" <<'PYEOF'
import json, sys
state = json.load(open(sys.argv[1]))
for e in state:
    if e["status"] != "success":
        print(e["index"])
        sys.exit(0)
print(len(state))
PYEOF
}

# ---------------------------------------------------------------------------
# Time-slicing precondition check.
# Checks label AND capacity (AND the gpu.replicas label) in the same call to
# defeat the race condition observed during Phase A, where mig.config.state
# read "success" for a moment before capacity had actually updated.
# ---------------------------------------------------------------------------
check_timeslicing_state() {
  local node mig_label gpu_capacity replicas_label
  node=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
  mig_label=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config}')
  gpu_capacity=$(kubectl get node "$node" -o jsonpath='{.status.capacity.nvidia\.com/gpu}')
  replicas_label=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.replicas}')

  if [ "$mig_label" != "all-disabled" ]; then
    echo "HALT: mig.config expected 'all-disabled', got '$mig_label'"
    return 1
  fi
  if [ "$gpu_capacity" != "10" ]; then
    echo "HALT: nvidia.com/gpu capacity expected 10, got '$gpu_capacity'"
    return 1
  fi
  if [ "$replicas_label" != "10" ]; then
    echo "HALT: gpu.replicas label expected 10, got '$replicas_label'"
    return 1
  fi
  return 0
}

is_per_pod_metric() {
  local fname="$1" m
  for m in "${PER_POD_METRICS[@]}"; do
    [[ "$fname" == *"_${m}_"* ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Relaxed per-pod row validation.
# run_experiment_v3.py has no live pod-health monitoring during the 60-minute
# recording window (unlike v4), and a Deployment auto-restarts a crashed pod,
# which can fragment "sum by (pod)" series into extra pod identities. A tight
# 700-720*r band would false-positive-halt on pod churn that isn't a real
# failure, so this checks rows-per-pod instead of raw row count: too few
# rows/pod halts (something genuinely broken), too many rows/pod warns but
# passes (likely a pod restart, not a lost experiment). Applied universally
# across all 50 experiments, not just workloads expected to be at risk.
# ---------------------------------------------------------------------------
validate_per_pod_file() {
  local file="$1"
  local total_rows=$(($(wc -l < "$file") - 1))
  local distinct_pods=$(awk -F, 'NR>1 {print $NF}' "$file" | sort -u | grep -v '^$' | wc -l)
  if [ "$distinct_pods" -eq 0 ]; then
    echo "HALT: no distinct pods found in $file"
    return 1
  fi
  local rows_per_pod=$((total_rows / distinct_pods))
  if [ "$rows_per_pod" -lt 500 ]; then
    echo "HALT: $file has $rows_per_pod rows/pod (<500)"
    return 1
  fi
  if [ "$rows_per_pod" -gt 1080 ]; then
    echo "WARN: $file has $rows_per_pod rows/pod (>1080, likely pod restart)"
    PENDING_WARNINGS+=("$(basename "$file"): $rows_per_pod rows/pod (>1080)")
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------

if [ -z "${TMUX:-}" ]; then
  echo "WARNING: not running under tmux. This is a ~38 hour unattended batch;"
  echo "if this shell dies, the batch dies with it."
fi

# git push origin extension-h100 always pushes the LOCAL branch named
# extension-h100, regardless of which branch is checked out. Guard against
# running this from the wrong branch and silently pushing unrelated local
# commits from extension-h100 to the real remote (bit us during dry-run
# testing on a throwaway branch).
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "extension-h100" ]; then
  echo "HALT: must be run from the extension-h100 branch (currently on $CURRENT_BRANCH)."
  exit 1
fi

echo "Checking time-slicing precondition..."
if ! check_timeslicing_state; then
  echo "HALT: time-slicing precondition failed (see above). Fix cluster state before running."
  exit 1
fi
echo "Time-slicing precondition OK: mig.config=all-disabled, capacity=10, gpu.replicas=10"

init_state

CRASHED=$(check_in_progress_crash) && CRASH_STATUS=0 || CRASH_STATUS=1
if [ "$CRASH_STATUS" -ne 0 ]; then
  echo "MANUAL INTERVENTION REQUIRED: crashed batch detected (in_progress entries):"
  echo "$CRASHED"
  echo "Set status to 'failed' for these entries in $STATE_FILE before resuming."
  exit 1
fi

START_INDEX=$(find_start_index)
TOTAL=$((${#WORKLOADS[@]} * ${#REPLICAS[@]}))
echo "Resuming from index $START_INDEX of $TOTAL"

BATCH_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  echo ""
  echo "## Tier 3 batch started $BATCH_START — $TOTAL experiments (index $START_INDEX onward)"
} >> "$JOURNAL"

# ---------------------------------------------------------------------------
# Per-experiment loop
# ---------------------------------------------------------------------------

INDEX=0
for workload in "${WORKLOADS[@]}"; do
  POD_READY_TIMEOUT=$POD_READY_TIMEOUT_DEFAULT
  [ "$workload" = "whisper" ] && POD_READY_TIMEOUT=$POD_READY_TIMEOUT_WHISPER

  for r in "${REPLICAS[@]}"; do
    if [ "$INDEX" -lt "$START_INDEX" ]; then
      INDEX=$((INDEX + 1))
      continue
    fi
    INDEX=$((INDEX + 1))

    CURRENT_WORKLOAD="$workload"
    CURRENT_R="$r"

    echo ""
    echo "=========================================================="
    echo "  [$INDEX/$TOTAL] $workload r=$r"
    echo "=========================================================="

    mark_status "$workload" "$r" "in_progress"

    bash tools/pre_experiment_checklist.sh || halt "pre_experiment_checklist failed"
    bash tools/clear_system_cache.sh || halt "clear_system_cache failed"

    LOGFILE="/tmp/tier3_${workload}_r${r}.log"
    nohup env PROMETHEUS_URL="$PROM_URL" DATA_OUTPUT_DIR="$DATA_DIR" \
      EXPERIMENT_AUTO_CONFIRM=1 PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" tools/run_experiment_v3.py \
      "$workload" "$r" > "$LOGFILE" 2>&1
    wait
    RUNNER_EXIT=$?

    [ "$RUNNER_EXIT" -eq 0 ] || halt "runner exit $RUNNER_EXIT (see $LOGFILE)"
    grep -q "EXPERIMENT COMPLETE" "$LOGFILE" || halt "no success marker in $LOGFILE"

    DIR="$DATA_DIR/${workload}_r${r}"
    [ -d "$DIR" ] || halt "output dir missing: $DIR"

    FILE_COUNT=$(ls "$DIR"/*.csv 2>/dev/null | wc -l)
    [ "$FILE_COUNT" -eq "$EXPECTED_FILE_COUNT" ] || halt "expected $EXPECTED_FILE_COUNT csv files, got $FILE_COUNT in $DIR"
    ls "$DIR"/*_timestamps.txt >/dev/null 2>&1 || halt "timestamps.txt missing in $DIR"

    TOTAL_ROWS=0
    PENDING_WARNINGS=()
    for f in "$DIR"/*.csv; do
      FNAME=$(basename "$f")
      ROWS=$(($(wc -l < "$f") - 1))
      TOTAL_ROWS=$((TOTAL_ROWS + ROWS))

      if is_per_pod_metric "$FNAME"; then
        validate_per_pod_file "$f" || halt "per-pod row validation failed: $f"
      else
        if [ "$ROWS" -lt "$EXPECTED_ROWS_MIN" ] || [ "$ROWS" -gt "$EXPECTED_ROWS_MAX" ]; then
          halt "row count out of range in $f: $ROWS (expected $EXPECTED_ROWS_MIN-$EXPECTED_ROWS_MAX)"
        fi
      fi

      # Null-cell check on the 'value' column. Tier 3 has no per-slice files
      # (whole-GPU, no MIG), so this applies uniformly to every CSV.
      EMPTY_VALUES=$(awk -F',' 'NR>1 && $2==""' "$f" | wc -l)
      if [ "$EMPTY_VALUES" -gt 0 ]; then
        halt "null value cell(s) in $f: $EMPTY_VALUES row(s)"
      fi
    done
    MEAN_ROWS=$((TOTAL_ROWS / FILE_COUNT))

    sleep 10
    PODS=$(kubectl get pods -l app="$workload" --no-headers 2>/dev/null | wc -l)
    [ "$PODS" -eq 0 ] || halt "stragglers: $PODS pods remain after $workload r=$r"

    check_timeslicing_state || halt "time-slicing state changed during $workload r=$r"

    set_csv_paths "$workload" "$r" "$DIR"

    WARN_MSG=""
    if [ "${#PENDING_WARNINGS[@]}" -gt 0 ]; then
      WARN_MSG=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${PENDING_WARNINGS[@]}")
      echo "[WARN] $workload r=$r: ${#PENDING_WARNINGS[@]} row-count warning(s) — see $STATE_FILE error_message"
    fi
    mark_status "$workload" "$r" "success" "$WARN_MSG"

    echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ) Tier 3 $workload r=$r: $FILE_COUNT CSV files, mean rows=$MEAN_ROWS" >> "$JOURNAL"

    # STATE_FILE is intentionally NOT committed (gitignored — machine-local
    # resume checkpoint). Committing $DIR and $JOURNAL only.
    git add "$DIR" "$JOURNAL"

    # Defensive: verify git add actually staged files from $DIR.
    # If $DIR is gitignored by mistake, git add silently skips its
    # contents and only the journal line stages, producing a data-less
    # commit. Halt loudly rather than pushing 50 empty commits.
    STAGED_FROM_DIR=$(git diff --cached --name-only | grep -c "^${DIR}/" || true)
    if [ "$STAGED_FROM_DIR" -eq 0 ]; then
      halt "git add staged 0 files from $DIR (path may be gitignored)"
    fi

    git commit -m "data: H100 Tier 3 $workload r=$r collected on devLab (time-slicing 10)" || halt "git commit failed"
    git push origin extension-h100 || halt "git push failed"

    echo "[OK] $workload r=$r complete ($FILE_COUNT csv files, mean rows=$MEAN_ROWS)"
  done
done

echo ""
echo "=========================================================="
echo "  TIER 3 BATCH COMPLETE: $TOTAL/$TOTAL experiments"
echo "=========================================================="
