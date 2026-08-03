#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-${ROOT}/data/working-state-labels/scale1000_rollouts_retry1}"
EXCLUDE_QIDS_FILE="${EXCLUDE_QIDS_FILE:-${RUN_DIR}/existing_qids.json}"
PYTHON="${PYTHON:-${ROOT}/vendor/openresearcher/.venv/bin/python}"
mkdir -p "$RUN_DIR"

pools=(
  data/working-state-labels/working_state_retrospective_mimo25pro_pool12.jsonl
  data/working-state-labels/working_state_retrospective_multiloop_pool8.jsonl
  data/working-state-labels/working_state_retrospective_mimo25pro_expand10g.jsonl
  data/working-state-labels/working_state_retrospective_mimo25pro_expand15d.jsonl
  data/working-state-labels/working_state_retrospective_mimo25pro_expand15e.jsonl
  data/working-state-labels/working_state_retrospective_mimo25pro_expand15f.jsonl
  data/working-state-labels/working_state_retrospective_mimo25pro_expand15h.jsonl
  data/working-state-labels/working_state_retrospective_mimo25pro_replacements6.jsonl
)
for shard in 0 1 2 3; do
  pools+=("data/working-state-labels/scale200/working_state_retrospective_mimo25pro_shard${shard}.jsonl")
done
for shard in $(seq 0 31); do
  path="data/working-state-labels/scale1000/working_state_retrospective_mimo25pro_shard${shard}.jsonl"
  if [[ -f "$path" ]]; then
    pools+=("$path")
  fi
done
for shard in $(seq 0 15); do
  path="data/working-state-labels/scale1000_rollouts/working_state_retrospective_mimo25pro_shard${shard}.jsonl"
  if [[ -f "$path" ]]; then
    pools+=("$path")
  fi
done

"$PYTHON" - "$EXCLUDE_QIDS_FILE" "${pools[@]}" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
qids = set()
for raw_path in sys.argv[2:]:
    with Path(raw_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                qids.add(str(json.loads(line)["source"]["qid"]))
output.write_text(
    json.dumps(
        sorted(
            qids,
            key=lambda value: (
                not value.isdigit(),
                int(value) if value.isdigit() else value,
            ),
        ),
        indent=2,
    ),
    encoding="utf-8",
)
print(f"excluded_existing_qids={len(qids)}")
PY

ALREADY_NEW=$("$PYTHON" - <<'PY'
from pathlib import Path

print(sum(
    sum(1 for line in path.open(encoding="utf-8") if line.strip())
    for root in (
        Path("data/working-state-labels/scale1000"),
        Path("data/working-state-labels/scale1000_rollouts"),
    )
    for path in root.glob("*.segments.jsonl")
))
PY
)
TOTAL_TARGET_TRAJECTORIES=$((800 - ALREADY_NEW))
if (( TOTAL_TARGET_TRAJECTORIES <= 0 )); then
  echo "The 800 new trajectories are already complete."
  exit 0
fi
echo "already_new_trajectories=${ALREADY_NEW} remaining=${TOTAL_TARGET_TRAJECTORIES}"

sbatch --parsable \
  --array="0-15%2" \
  --export="ALL,PROJECT_ROOT=${ROOT},RUN_DIR=${RUN_DIR},EXCLUDE_QIDS_FILE=${EXCLUDE_QIDS_FILE},NUM_SHARDS=16,TOTAL_TARGET_TRAJECTORIES=${TOTAL_TARGET_TRAJECTORIES},DATA_MODEL=mimo-v2.5-pro" \
  scripts/slurm/build_working_state_scale1000.sbatch
