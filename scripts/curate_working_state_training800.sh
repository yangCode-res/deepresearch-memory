#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUTPUT="${CONTROLLER_OUTPUT:-data/working-state-labels/working_state_retrospective_mimo25pro_training800.jsonl}"

args=(scripts/curate_retrospective_training20.py)
for stem in \
  working_state_retrospective_mimo25pro_pool12 \
  working_state_retrospective_multiloop_pool8 \
  working_state_retrospective_mimo25pro_expand10g \
  working_state_retrospective_mimo25pro_expand15d \
  working_state_retrospective_mimo25pro_expand15e \
  working_state_retrospective_mimo25pro_expand15f \
  working_state_retrospective_mimo25pro_expand15h \
  working_state_retrospective_mimo25pro_replacements6; do
  args+=(--pool "data/working-state-labels/${stem}.jsonl")
  args+=(--segments "data/working-state-labels/${stem}.segments.jsonl")
done
for shard in 0 1 2 3; do
  stem="data/working-state-labels/scale200/working_state_retrospective_mimo25pro_shard${shard}"
  args+=(--pool "${stem}.jsonl" --segments "${stem}.segments.jsonl")
done
args+=(
  --output "$OUTPUT"
  --questions 200
  --min-positive-retrieval 150
  --contract-overrides configs/working_state_training200_contract_overrides.json
)

exec "$PYTHON" "${args[@]}"
