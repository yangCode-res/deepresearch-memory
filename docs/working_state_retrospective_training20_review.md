# Retrospective Working-State training20 review

## Verdict

`working_state_retrospective_training20.jsonl` is accepted as a small, training-ready pilot set.
It is suitable for validating the Working-State / Loop-decision learning contract. It is not large
enough to establish production model quality.

The labels are behavioral retrospective annotations: Loop boundaries follow what the recorded
OpenResearcher agent actually did next. They are not counterfactual optimal-policy labels and should
not be used alone to teach early stopping.

## Construction

- Teacher model: `mimo-v2.5`
- Complete candidate trajectories: 8 unique qids
- Candidate decisions replayed: 54
- Candidate distribution: 34 CONTINUE / 12 SWITCH / 8 READY
- Teacher requests: 68
- Teacher tokens: 288,756
- Incomplete trajectories committed: 0
- Teacher/JSON failures: 0
- Causal message-ID violations: 0

Five complete multi-Loop trajectories were selected. Each contributes exactly four independent
training points:

1. one CONTINUE immediately before the first accepted boundary;
2. the SWITCH boundary;
3. one CONTINUE after the switch that retrieves prior Loop memory;
4. the recorded READY boundary.

The resulting distribution is exactly 10 CONTINUE / 5 SWITCH / 5 READY, with four samples from each
of five unique qids.

## Manual boundary review

| QID | Accepted boundary | Why it is a distinct information dependency | Result |
| --- | --- | --- | --- |
| 5605 | establish the NaHCO3 mass calculation → independently verify the mEq/mg conversion factor | The numerical solution and the conversion rule can be established independently; the second is a verification dependency used by the recorded agent. | PASS |
| 5511 | identify the award-matching person → eliminate the explicitly offered alternative candidate | The user supplied two candidates. Positive identification and alternative-candidate elimination are separately decidable. | PASS |
| 6474 | identify the referenced study passage → extract and verify the exact date | Artifact/passage localization is complete before the requested fact is extracted from that passage. | PASS |
| 6272 | identify the specified Frontiers article → extract the requested worker-task percentage | Correct-artifact localization and requested-statistic extraction are separately observable stages. | PASS |
| 6382 | identify the specified article and relevant section → extract the DOAJ paper count | Correct-artifact localization and requested-number extraction are separately observable stages. | PASS |

Rejected patterns observed in the wider candidate pool included changing to another source for the
same claim, collecting a stronger citation for an already established answer, and formatting citation
lines. Those are evidence-acquisition strategies inside one information dependency, not new Loops.

## READY review

The original parquet trajectories were re-opened for qids 5605, 5511, 6474, 6272, and 6382. In every
case the final selected tool-result decision is followed by answer synthesis and an assistant `final`
message, with no later research tool call. All five READY labels therefore match recorded terminal
transitions.

## Memory review

- All five selected post-switch CONTINUE samples retrieve `memory_loop_001`.
- QID 5605's READY sample retrieves both `memory_loop_001` and `memory_loop_002`.
- A closed-Loop memory appears in the next full-replay input before it can be selected.
- CONTINUE samples use `NOOP` StateDelta and emit no cross-Loop memory.
- SWITCH and READY samples use `APPLY` and emit causal, evidence-cited handoff memory.

## Leakage and contract review

The final set passes all of the following:

- no message after `prefix_end_message_index` appears in model input;
- no target cites a future `msg_NNNN` coordinate;
- retrospective segmentation and next-Loop contracts are absent from training input;
- Working-State and Global-State replay chains are continuous in the complete pool;
- directional contracts contain no URL, domain, browser operation, query, source, Doc ID, or search-result instruction;
- answer-like dates, percentages, and numbers not visible when a Loop becomes active are scrubbed;
- CONTINUE preserves its committed Loop contract and information gain does not increase inside a Loop;
- selected records pass the project schema validator with zero violations and zero subgoal drift.

During review, two defects were caught and fixed before acceptance:

1. a candidate contained “view Doc … / find another source” in directional state;
2. qid 6272 leaked `15%` into a completion test before that value was observed. The numeric scrubber
   had incorrectly treated the `15` in line labels as visibility for `15%`; percentage tokens are now
   matched atomically.

## Artifacts

- `data/working-state-labels/working_state_retrospective_training20.jsonl`
- `data/working-state-labels/working_state_retrospective_training20.preview.md`
- `data/working-state-labels/working_state_retrospective_training20.segments.jsonl`
- `data/working-state-labels/working_state_retrospective_training20.qa.json`

