#!/bin/bash
# Unattended orchestration of 35 Tier 2 experiments (5 workloads x r=1..7).
#
# Invocation:
#   tmux new -s tier2-batch
#   nohup bash tools/run_tier2_batch.sh > /tmp/tier2_batch.log 2>&1 &
#   # detach with Ctrl-B D
set -uo pipefail

WORKLOADS=(bert gpt2 resnet152 whisper yolo)
REPLICAS=(1 2 3 4 5 6 7)
PROM_URL="${PROM_URL:-http://172.22.174.66:30090}"
DATA_DIR="${DATA_DIR:-data/raw/extension_tier2}"
STATE_FILE=".tier2_state.json"
JOURNAL="EXTENSION_JOURNAL.md"
POD_READY_TIMEOUT_DEFAULT=180
POD_READY_TIMEOUT_WHISPER=300
CLEANUP_TIMEOUT=60
EXPECTED_ROWS_MIN=700
EXPECTED_ROWS_MAX=720
EXPECTED_FILE_COUNT=33
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
        if status == "failed" and error_msg:
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
# MIG precondition check.
# Step 0 confirmed mig.config.state is surfaced as a node LABEL (not an
# annotation) on this cluster — hardcoded to that surface per that finding.
# ---------------------------------------------------------------------------
check_mig_state() {
  local node cfg state cap
  node=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
  cfg=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config}')
  state=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config\.state}')
  cap=$(kubectl get node "$node" -o jsonpath='{.status.capacity.nvidia\.com/gpu}')

  if [ "$cfg" != "all-1g.12gb" ]; then
    echo "MIG check failed: mig.config=$cfg (expected all-1g.12gb)"
    return 1
  fi
  if [ "$state" != "success" ]; then
    echo "MIG check failed: mig.config.state=$state (expected success)"
    return 1
  fi
  if [ "$cap" != "7" ]; then
    echo "MIG check failed: nvidia.com/gpu capacity=$cap (expected 7)"
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

echo "Checking MIG precondition..."
if ! check_mig_state; then
  echo "HALT: MIG precondition failed (see above). Fix cluster state before running."
  exit 1
fi
echo "MIG precondition OK: all-1g.12gb, success, 7 slices allocatable"

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
  echo "## Tier 2 batch started $BATCH_START — $TOTAL experiments (index $START_INDEX onward)"
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

    LOGFILE="/tmp/tier2_${workload}_r${r}.log"
    nohup env PROMETHEUS_URL="$PROM_URL" DATA_OUTPUT_DIR="$DATA_DIR" \
      EXPERIMENT_AUTO_CONFIRM=1 \
      "$PYTHON_BIN" tools/run_experiment_v4.py \
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
    for f in "$DIR"/*.csv; do
      FNAME=$(basename "$f")
      ROWS=$(($(wc -l < "$f") - 1))
      TOTAL_ROWS=$((TOTAL_ROWS + ROWS))

      if [[ "$FNAME" == *"_per_slice_"* ]]; then
        # One row per MIG slice (7) per tick, regardless of replica count.
        MIN=$((EXPECTED_ROWS_MIN * 7))
        MAX=$((EXPECTED_ROWS_MAX * 7))
      elif is_per_pod_metric "$FNAME"; then
        # One row per running pod per tick.
        MIN=$((EXPECTED_ROWS_MIN * r))
        MAX=$((EXPECTED_ROWS_MAX * r))
      else
        MIN=$EXPECTED_ROWS_MIN
        MAX=$EXPECTED_ROWS_MAX
      fi

      if [ "$ROWS" -lt "$MIN" ] || [ "$ROWS" -gt "$MAX" ]; then
        halt "row count out of range in $f: $ROWS (expected $MIN-$MAX)"
      fi

      # Null-cell check on aggregated (non-per-slice) files, scoped to the
      # 'value' column only. Aggregated GPU files intentionally write blank
      # pod/container/namespace/pci_bus_id cells by design, so a whole-line
      # blank-cell check would false-positive on every one of them; only a
      # missing measurement (empty value) indicates a real problem.
      if [[ "$FNAME" != *"_per_slice_"* ]]; then
        EMPTY_VALUES=$(awk -F',' 'NR>1 && $2==""' "$f" | wc -l)
        if [ "$EMPTY_VALUES" -gt 0 ]; then
          halt "null value cell(s) in $f: $EMPTY_VALUES row(s)"
        fi
      fi
    done
    MEAN_ROWS=$((TOTAL_ROWS / FILE_COUNT))

    sleep 10
    PODS=$(kubectl get pods -l app="$workload" --no-headers 2>/dev/null | wc -l)
    [ "$PODS" -eq 0 ] || halt "stragglers: $PODS pods remain after $workload r=$r"

    check_mig_state || halt "MIG state changed during $workload r=$r"

    set_csv_paths "$workload" "$r" "$DIR"
    mark_status "$workload" "$r" "success"

    echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ) Tier 2 $workload r=$r: $FILE_COUNT csv files, mean rows=$MEAN_ROWS" >> "$JOURNAL"

    # STATE_FILE is intentionally NOT committed (gitignored — machine-local
    # resume checkpoint). Committing $DIR and $JOURNAL only.
    git add "$DIR" "$JOURNAL"
    git commit -m "data: H100 Tier 2 $workload r=$r collected on devLab (MIG 1g.12gb)" || halt "git commit failed"
    git push origin extension-h100 || halt "git push failed"

    echo "[OK] $workload r=$r complete ($FILE_COUNT csv files, mean rows=$MEAN_ROWS)"
  done
done

echo ""
echo "=========================================================="
echo "  TIER 2 BATCH COMPLETE: $TOTAL/$TOTAL experiments"
echo "=========================================================="
