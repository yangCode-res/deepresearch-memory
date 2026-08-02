#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-${ROOT}/data/working-state-labels/scale200}"
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
    json.dumps(sorted(qids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)), indent=2),
    encoding="utf-8",
)
PY

sbatch --parsable \
  --array="0-3%2" \
  --export="ALL,PROJECT_ROOT=${ROOT},RUN_DIR=${RUN_DIR},EXCLUDE_QIDS_FILE=${EXCLUDE_QIDS_FILE},NUM_SHARDS=4,TARGET_TRAJECTORIES_PER_SHARD=30,DATA_MODEL=mimo-v2.5-pro" \
  scripts/slurm/build_working_state_scale200.sbatch
