#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUTPUT="${RESEARCHER_OUTPUT:-data/researcher-training/researcher_memory_conditioned.jsonl}"

args=(
  scripts/build_researcher_memory_conditioned.py
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_pool12.jsonl
  --pool data/working-state-labels/working_state_retrospective_multiloop_pool8.jsonl
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_expand10g.jsonl
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_expand15d.jsonl
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_expand15e.jsonl
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_expand15f.jsonl
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_expand15h.jsonl
  --pool data/working-state-labels/working_state_retrospective_mimo25pro_replacements6.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_pool12.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_multiloop_pool8.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_expand10g.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_expand15d.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_expand15e.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_expand15f.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_expand15h.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_mimo25pro_replacements6.segments.jsonl
  --segments data/working-state-labels/working_state_retrospective_training20.segments.jsonl
  --controller-training data/working-state-labels/working_state_retrospective_mimo25pro_training200.jsonl
  --contract-overrides configs/working_state_training200_contract_overrides.json
  --output "$OUTPUT"
)

if [[ -n "${OPENRESEARCHER_RAW_GLOB:-}" ]]; then
  args+=(--raw-input-glob "$OPENRESEARCHER_RAW_GLOB")
fi

exec "$PYTHON" "${args[@]}"
