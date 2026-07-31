"""Unified API controller for loop, state, and memory-selection decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .llm_planner import OpenAIChatJSONClient
from .retrieval import MemoryHit
from .schema import GlobalIntentState, TaskAnchor
from .trajectory import TrajectoryEvent


TASK_STATUSES = {"CONTINUE", "SWITCH_LOOP", "READY_TO_ANSWER"}
RESEARCH_PHASES = {
    "DISCOVERY",
    "CANDIDATE_VERIFICATION",
    "EVIDENCE_COMPLETION",
    "ANSWER_SYNTHESIS",
}
LOOP_OUTCOMES = {"IN_PROGRESS", "RESOLVED", "REFUTED", "BLOCKED", "SUPERSEDED"}
BOUNDARY_BASES = {
    "NONE",
    "SUBGOAL_COMPLETED",
    "SUBGOAL_CHANGED",
    "CANDIDATE_CHANGED",
    "BLOCKED_OR_SATURATED",
    "PHASE_TRANSITION",
    "TASK_COMPLETE",
}
VALID_ITEM_STATUSES = {"active", "tentative", "confirmed", "rejected", "superseded", "resolved"}
VALID_SOURCE_TYPES = {"user", "agent", "tool", "paper", "web", "experiment", "system"}
INFORMATION_GAIN_LEVELS = {"HIGH", "MEDIUM", "LOW"}


def _canonical_subgoal(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def empty_loop_progress() -> dict[str, Any]:
    """Create ephemeral, current-loop-only progress state."""

    return {
        "completion_test": "",
        "progress_summary": "",
        "resolved_aspects": [],
        "open_aspects": [],
        "key_evidence": [],
        "candidate_answer": "",
        "answer_stable": False,
        "evidence_sufficient": False,
        "confidence": 0.0,
        "expected_information_gain": "HIGH",
        "tried_strategies": [],
        "rejected_hypotheses": [],
        "blocked_subgoals": [],
        "promising_leads": [],
        "prioritized_open_aspects": [],
        "research_direction": {
            "objective": "",
            "must_investigate": [],
            "rationale": "",
            "stop_condition": "",
        },
        "avoid": [],
    }


def _loop_progress_output_contract() -> dict[str, Any]:
    contract = empty_loop_progress()
    contract.update(
        {
            "tried_strategies": [
                {
                    "strategy": "semantic strategy family",
                    "outcome": "compact outcome",
                    "evidence_gain": "HIGH|MEDIUM|LOW|NONE",
                }
            ],
            "promising_leads": [
                {
                    "kind": "ENTITY|URL|RESULT",
                    "entity": "concrete named entity, URL, or result ID",
                    "source": "where this lead appeared",
                    "reason": "why it matches task clues",
                    "status": "ACTIVE|VERIFIED|REJECTED",
                    "confidence": 0.8,
                }
            ],
            "prioritized_open_aspects": [
                {
                    "aspect": "unresolved aspect",
                    "priority": "ANSWER_CRITICAL|CORROBORATING",
                    "status": "open|resolved",
                    "best_next_action": "direction only, no tool or exact query",
                }
            ],
            "research_direction": {
                "objective": "current directional objective",
                "must_investigate": ["concrete lead from promising_leads"],
                "rationale": "why this direction has value",
                "stop_condition": "observable condition for closing this direction",
            },
        }
    )
    return contract


def _normalize_loop_progress(raw: Any, previous: dict[str, Any] | None) -> dict[str, Any]:
    prior = {**empty_loop_progress(), **(previous or {})}
    source = raw if isinstance(raw, dict) else {}
    normalized = dict(prior)
    for name, limit in (("completion_test", 600), ("progress_summary", 1000), ("candidate_answer", 500)):
        if name in source:
            normalized[name] = str(source.get(name) or "")[:limit]
    for name, count, limit in (
        ("resolved_aspects", 12, 500),
        ("open_aspects", 12, 500),
        ("key_evidence", 12, 800),
    ):
        value = source.get(name)
        if isinstance(value, list):
            normalized[name] = [str(item)[:limit] for item in value if str(item).strip()][:count]
    for name in ("answer_stable", "evidence_sufficient"):
        if isinstance(source.get(name), bool):
            normalized[name] = source[name]
    if "confidence" in source:
        try:
            normalized["confidence"] = max(0.0, min(1.0, float(source["confidence"])))
        except (TypeError, ValueError):
            pass
    gain = str(source.get("expected_information_gain") or "").upper()
    if gain in INFORMATION_GAIN_LEVELS:
        normalized["expected_information_gain"] = gain

    def compact_records(name: str, fields: tuple[str, ...], count: int = 8) -> None:
        value = source.get(name)
        if not isinstance(value, list):
            return
        records: list[dict[str, str]] = []
        for item in value[:count]:
            if not isinstance(item, dict):
                continue
            record = {
                field_name: str(item.get(field_name) or "")[:500]
                for field_name in fields
            }
            if any(record.values()):
                records.append(record)
        normalized[name] = records

    compact_records("tried_strategies", ("strategy", "outcome", "evidence_gain"))
    # Promising leads are sticky: a named entity, URL, or concrete result must
    # not disappear just because a later controller response omits it. Generic
    # suggestions such as "search for other brands" are directions, not leads.
    prior_leads = prior.get("promising_leads") or []
    incoming_leads = source.get("promising_leads") or []
    merged_leads: dict[tuple[str, str], dict[str, Any]] = {}
    for item in [*prior_leads, *incoming_leads]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").upper()
        entity = str(item.get("entity") or "").strip()[:500]
        source_ref = str(item.get("source") or "").strip()[:500]
        status = str(item.get("status") or "ACTIVE").upper()
        if kind not in {"ENTITY", "URL", "RESULT"} or not entity or not source_ref:
            continue
        if status not in {"ACTIVE", "VERIFIED", "REJECTED"}:
            status = "ACTIVE"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        record = {
            "kind": kind,
            "entity": entity,
            "source": source_ref,
            "reason": str(item.get("reason") or "")[:500],
            "status": status,
            "confidence": confidence,
        }
        key = (kind, entity.casefold())
        previous_record = merged_leads.get(key)
        if previous_record is None or confidence >= float(previous_record.get("confidence", 0.0)):
            merged_leads[key] = record
    normalized["promising_leads"] = list(merged_leads.values())[:8]
    compact_records(
        "prioritized_open_aspects",
        ("aspect", "priority", "status", "best_next_action"),
        count=12,
    )
    for name in ("rejected_hypotheses", "blocked_subgoals", "avoid"):
        value = source.get(name)
        if isinstance(value, list):
            sticky = [str(item)[:500] for item in prior.get(name) or [] if str(item).strip()]
            sticky.extend(str(item)[:500] for item in value if str(item).strip())
            normalized[name] = list(dict.fromkeys(sticky))[:12]

    direction = source.get("research_direction")
    if isinstance(direction, dict):
        normalized_direction = dict(prior.get("research_direction") or {})
        for name in ("objective", "rationale", "stop_condition"):
            if name in direction:
                normalized_direction[name] = str(direction.get(name) or "")[:600]
        must_investigate = direction.get("must_investigate")
        if isinstance(must_investigate, list):
            normalized_direction["must_investigate"] = [
                str(item)[:500] for item in must_investigate if str(item).strip()
            ][:5]
        normalized["research_direction"] = normalized_direction
    return normalized


def _ensure_research_direction(
    raw: dict[str, Any], *, stagnant_rounds: int
) -> None:
    """Supply directional guidance without taking tool choice from the researcher."""

    loop = raw.get("loop") if isinstance(raw.get("loop"), dict) else {}
    progress = loop.get("progress") if isinstance(loop.get("progress"), dict) else {}
    direction = progress.get("research_direction")
    if not isinstance(direction, dict):
        direction = {}
    status = str(raw.get("task_status") or "CONTINUE").upper()
    if status != "READY_TO_ANSWER":
        subgoal_value = (
            loop.get("next_loop_subgoal")
            if status == "SWITCH_LOOP"
            else loop.get("current_loop_subgoal")
        )
        subgoal = str(subgoal_value or "").strip()
        defaults = {
            "objective": subgoal,
            "must_investigate": [],
            "rationale": str(loop.get("reason") or ""),
            "stop_condition": str(
                progress.get("completion_test")
                or "obtain evidence that resolves the current subgoal"
            ),
        }
        for name, value in defaults.items():
            if name == "must_investigate":
                if not isinstance(direction.get(name), list):
                    direction[name] = value
            elif not str(direction.get(name) or "").strip():
                direction[name] = value
        if not direction.get("must_investigate"):
            active_leads = sorted(
                (
                    item
                    for item in progress.get("promising_leads") or []
                    if isinstance(item, dict) and item.get("status") == "ACTIVE"
                ),
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )
            direction["must_investigate"] = [
                str(item.get("entity") or "") for item in active_leads[:3]
                if str(item.get("entity") or "").strip()
            ]
    progress["research_direction"] = direction
    if stagnant_rounds >= 2 and not progress.get("avoid"):
        tried = progress.get("tried_strategies") or []
        latest_strategy = ""
        if tried and isinstance(tried[-1], dict):
            latest_strategy = str(tried[-1].get("strategy") or "").strip()
        progress["avoid"] = [
            latest_strategy or "semantic reformulations of the latest no-gain search strategy"
        ]
    loop["progress"] = progress


@dataclass
class UnifiedControlDecision:
    task_status: str = "CONTINUE"
    research_phase: str = "DISCOVERY"
    switch_loop: bool = False
    reason: str = ""
    confidence: float = 0.0
    current_loop_subgoal: str = ""
    next_loop_subgoal: str = ""
    loop_outcome: str = "IN_PROGRESS"
    boundary_basis: str = "NONE"
    loop_progress: dict[str, Any] = field(default_factory=empty_loop_progress)
    state_delta: dict[str, Any] = field(default_factory=lambda: {"summary": "", "operations": []})
    retrieval_query: str = ""
    selected_memory_ids: list[str] = field(default_factory=list)
    validation_retries: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)


def _event(event: TrajectoryEvent, limit: int = 1000) -> dict[str, Any]:
    return {"sequence": event.sequence, "role": event.role, "text": event.text[:limit]}


def _state(state: GlobalIntentState) -> dict[str, Any]:
    def values(name: str) -> list[Any]:
        return [
            {
                "id": item.item_id,
                "value": item.value,
                "status": item.status.value,
                "confidence": item.confidence,
            }
            for item in getattr(state, name)[-6:]
        ]
    return {
        "version": state.state_version,
        "current_goal": (
            {
                "id": state.current_goal.item_id,
                "value": state.current_goal.value,
                "status": state.current_goal.status.value,
                "confidence": state.current_goal.confidence,
            }
            if state.current_goal
            else None
        ),
        "active_subgoals": values("active_subgoals"),
        "working_hypotheses": values("working_hypotheses"),
        "resolved_findings": values("resolved_findings"),
        "open_questions": values("open_questions"),
        "uncertainties": values("uncertainties"),
        "next_actions": values("next_actions"),
    }


class UnifiedMemoryController:
    """Make all control-plane decisions in one structured API call per round."""

    def __init__(self, client: OpenAIChatJSONClient, *, max_selected_memories: int = 4) -> None:
        self.client = client
        self.max_selected_memories = max_selected_memories

    def decide(
        self,
        *,
        anchor: TaskAnchor,
        state: GlobalIntentState,
        current_loop: list[TrajectoryEvent],
        latest_events: list[TrajectoryEvent],
        candidates: list[MemoryHit],
        current_phase: str = "DISCOVERY",
        current_loop_subgoal: str = "",
        loop_progress: dict[str, Any] | None = None,
        loop_rounds: int = 0,
        stagnant_rounds: int = 0,
    ) -> UnifiedControlDecision:
        current_phase = str(current_phase or "DISCOVERY").upper()
        if current_phase not in RESEARCH_PHASES:
            raise ValueError(f"invalid current_phase {current_phase}")
        allowed_ids = {hit.memory_id for hit in candidates}
        system_prompt = (
            "You are the control plane for a general deep-research memory system. Return only valid JSON. "
            "In one decision: (1) determine whether the next model call should continue the current research "
            "work unit, start a new work unit, or stop researching and answer; (2) if the current work unit "
            "ends, describe the minimal durable StateDelta it produced; (3) select prior memories useful for "
            "the next model call. A loop is one coherent research work unit organized around one primary, "
            "locally decidable subgoal with a recognizable completion test. The subgoal may be identifying an "
            "entity, verifying one claim, resolving one dependency, comparing a defined set of options, "
            "testing one hypothesis, diagnosing one cause, or producing one bounded part of a deliverable. "
            "Tool choice, query wording, source choice, role changes, and intermediate reasoning do not define "
            "loop boundaries. Continue while the primary subgoal and its completion test remain the same. "
            "End the loop when that subgoal is resolved, refuted, explicitly abandoned after evidence "
            "saturation, superseded, or replaced by a different primary subgoal with a different completion "
            "test. Do not fragment a coherent evidence-gathering cycle, but do not combine independently "
            "decidable claims merely because they concern the same candidate or share a broad phase. "
            "Research phase is an orthogonal macro label, not the definition of a loop and not a prerequisite "
            "for switching loops. Classify the next call into exactly one phase: DISCOVERY means candidate or "
            "solution formation; CANDIDATE_VERIFICATION means testing a concrete candidate or hypothesis; "
            "EVIDENCE_COMPLETION means the answer is stable and only source or citation coverage is missing; "
            "ANSWER_SYNTHESIS means research is complete and the next call must answer. "
            "Never solve the user's research question yourself and never invent memory IDs or evidence IDs. "
            "The current-loop Working State is a directional policy, not a tool planner. Track failed "
            "search strategies, rejected hypotheses, promising entities or URLs visible in tool results, "
            "prioritized unresolved aspects, a research objective, must-investigate leads, a stop condition, "
            "and explicit directions to avoid. Never choose the research model's tool, URL target, exact "
            "query, or query wording. The research model owns those concrete choices. "
            "Operation contract: ADD may target active_subgoals, confirmed_constraints, soft_preferences, "
            "candidate_options, rejected_options, working_hypotheses, resolved_findings, open_questions, "
            "uncertainties, or next_actions. UPDATE may target current_goal only. RESOLVE may target "
            "open_questions only. Use mode=NOOP with operations=[] when a closed loop contains no durable "
            "state change. Never put citations or external document IDs in evidence_ids; leave it empty."
        )
        payload = {
            "task_anchor": {
                "goal": anchor.original_goal,
                "success_criteria": anchor.success_criteria,
                "immutable_constraints": anchor.immutable_constraints,
            },
            "global_state": _state(state),
            "current_research_phase": current_phase,
            "committed_current_loop_subgoal": current_loop_subgoal,
            "committed_loop_progress": _normalize_loop_progress(loop_progress, None),
            "loop_runtime": {
                "rounds_in_current_loop": max(0, int(loop_rounds)),
                "rounds_without_material_progress": max(0, int(stagnant_rounds)),
            },
            "current_loop": [_event(event) for event in current_loop[-30:]],
            "latest_events": [_event(event) for event in latest_events],
            "memory_candidates": [
                {
                    "id": hit.memory_id,
                    "type": hit.memory_type,
                    "source_type": hit.metadata.get("source_type"),
                    "text": hit.text[:1200],
                }
                for hit in candidates
            ],
            "required_output": {
                "task_status": "CONTINUE|SWITCH_LOOP|READY_TO_ANSWER",
                "research_phase": "DISCOVERY|CANDIDATE_VERIFICATION|EVIDENCE_COMPLETION|ANSWER_SYNTHESIS",
                "loop": {
                    "switch": False,
                    "reason": "short reason",
                    "confidence": 0.8,
                    "current_loop_subgoal": "short label",
                    "next_loop_subgoal": "short label",
                    "outcome": "IN_PROGRESS|RESOLVED|REFUTED|BLOCKED|SUPERSEDED",
                    "boundary_basis": "NONE|SUBGOAL_COMPLETED|SUBGOAL_CHANGED|CANDIDATE_CHANGED|BLOCKED_OR_SATURATED|PHASE_TRANSITION|TASK_COMPLETE",
                    "progress": _loop_progress_output_contract(),
                },
                "state_delta": {
                    "mode": "APPLY|NOOP",
                    "summary": "short summary; empty when switch=false",
                    "operations": [
                        {
                            "operation": "ADD|UPDATE|RESOLVE",
                            "target": "working_hypotheses|resolved_findings|open_questions|uncertainties|next_actions|active_subgoals",
                            "value": "concise value",
                            "reason": "why",
                            "evidence_ids": [],
                            "target_item_ids": [],
                            "item": {
                                "status": "tentative",
                                "confidence": 0.7,
                                "source_type": "agent",
                                "valid_time": None,
                                "contradicts": [],
                                "supersedes": [],
                                "user_confirmed": False,
                            },
                        }
                    ],
                },
                "retrieval": {
                    "query": "short semantic memory query",
                    "selected_memory_ids": [],
                    "reason": "short reason",
                },
            },
            "rules": [
                "First infer the primary subgoal and its completion test; judge the boundary from that work-unit contract, not from topic words or tool-call count.",
                "committed_current_loop_subgoal is authoritative. Treat it as the current work-unit contract; only loop.next_loop_subgoal may propose the next commitment.",
                "Update loop.progress every round. It is ephemeral working memory for the current loop, not Global State. Preserve previously verified aspects and citation-bearing key evidence unless new evidence refutes them.",
                "Maintain tried_strategies as semantic strategy families, not a raw query log. Mark evidence_gain as HIGH, MEDIUM, LOW, or NONE and record the outcome compactly.",
                "Extract promising_leads from the latest tool result even when the research model ignored them. A lead must have kind ENTITY, URL, or RESULT; a concrete entity; a source/result reference; status ACTIVE, VERIFIED, or REJECTED; confidence; and why it matches. Never store an instruction such as 'search for X' as a lead.",
                "Maintain rejected_hypotheses and avoid so the next model call does not revisit disproven regions, candidates, or semantically equivalent failed queries unless genuinely new evidence justifies it.",
                "Maintain blocked_subgoals. A subgoal closed as BLOCKED must not be selected again unless new evidence explicitly reopens it.",
                "prioritized_open_aspects must distinguish ANSWER_CRITICAL from CORROBORATING aspects and give the best next action for important unresolved aspects.",
                "For CONTINUE or SWITCH_LOOP, research_direction must give the objective, up to five must-investigate concrete leads, rationale, and observable stop condition. It must not contain a recommended tool, URL target, exact search query, or query wording.",
                "completion_test states what observable result closes this work unit. resolved_aspects and open_aspects form a compact coverage ledger; key_evidence stores only decisive citation-bearing facts, not search narration.",
                "Set answer_stable=true only when remaining open aspects cannot reasonably change candidate_answer. Set evidence_sufficient=true when core success criteria have citable support; corroborating clues may retain disclosed uncertainty.",
                "expected_information_gain estimates whether another search under the same subgoal is likely to change the answer or materially improve core evidence.",
                "Use loop_runtime to detect unproductive repetition. Several rounds without material ledger changes require either a genuinely different strategy/subgoal, a BLOCKED outcome, or READY_TO_ANSWER when the answer is already stable.",
                "When rounds_without_material_progress >= 2, prohibit the failed semantic strategy while leaving the concrete alternative to the research model. When it is >= 3, require investigation of the strongest active lead, switch subgoal, or answer; add the repeated approach to avoid.",
                "Return READY_TO_ANSWER when the answer is stable, the task's core success criteria have adequate citable support, unresolved details cannot reasonably change the exact answer, and further search has low expected information gain.",
                "Do not require equal-strength direct citations for every clue: distinguish identity-critical claims from corroborating clues and disclose residual uncertainty in the final answer.",
                "Return SWITCH_LOOP when the completed/current work unit is terminal and the next call will pursue a different primary subgoal or completion test. The research phase may stay the same.",
                "Return CONTINUE when the next action still serves the same primary subgoal and completion test, even if the query, source, tool, method, or intermediate hypothesis changes.",
                "READY_TO_ANSWER and CONTINUE require loop.switch=false; SWITCH_LOOP requires loop.switch=true.",
                "For CONTINUE use outcome=IN_PROGRESS and boundary_basis=NONE.",
                "For SWITCH_LOOP use a terminal outcome and the most specific non-NONE boundary_basis; provide distinct, non-empty current and next subgoal labels.",
                "A failed query alone is not BLOCKED. Use BLOCKED only after reasonable alternatives are exhausted and the next call intentionally moves to another dependency or strategy.",
                "For READY_TO_ANSWER use outcome=RESOLVED, boundary_basis=TASK_COMPLETE, research_phase=ANSWER_SYNTHESIS, and an empty next_loop_subgoal.",
                "Phase changes often coincide with a loop boundary, but phase equality never forbids a loop switch and phase change alone never proves one.",
                "Only emit StateDelta operations when loop.switch=true.",
                "When switch=false, state_delta.mode must be NOOP and operations must be empty.",
                "When switch=true, use APPLY with at least one concise operation that records the durable phase result.",
                "StateDelta values must be concise durable facts, never reasoning transcripts, search narration, or tool-call JSON.",
                "Preserve citation markers or source coordinates inside finding values when they are available.",
                "UPDATE+current_goal and RESOLVE+open_questions are the only valid non-ADD combinations.",
                "Use at most four StateDelta operations.",
                f"Select at most {self.max_selected_memories} candidate memory IDs.",
                "Prefer tool-source memories containing evidence and citations over assistant search narration.",
                "Return a single JSON object with exactly task_status, research_phase, loop, state_delta, and retrieval.",
            ],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False)
        raw: dict[str, Any] = {}
        last_error = ""
        for attempt in range(2):
            if attempt:
                repair = {
                    "validation_error": last_error,
                    "invalid_response": raw,
                    "instruction": "Return a corrected complete JSON object obeying the operation contract.",
                    "original_input": payload,
                }
                current_prompt = json.dumps(repair, ensure_ascii=False)
            else:
                current_prompt = user_prompt
            raw = self.client.complete_json(system_prompt, current_prompt)
            # Current-loop identity is system-owned state, not model-owned
            # output. Preserve the model's first label only when establishing
            # the initial commitment; afterwards overwrite any echo drift.
            if current_loop_subgoal and isinstance(raw.get("loop"), dict):
                raw["loop"]["current_loop_subgoal"] = current_loop_subgoal
            if isinstance(raw.get("loop"), dict):
                raw["loop"]["progress"] = _normalize_loop_progress(
                    raw["loop"].get("progress"), loop_progress
                )
            _ensure_research_direction(raw, stagnant_rounds=stagnant_rounds)
            retrieval_output = raw.get("retrieval")
            if isinstance(retrieval_output, dict) and isinstance(
                retrieval_output.get("selected_memory_ids"), list
            ):
                # Memory retrieval is advisory. An otherwise valid control
                # decision must not be discarded because the LLM echoed a
                # stale or invented memory ID; keep only IDs from this round's
                # candidate pool.
                retrieval_output["selected_memory_ids"] = [
                    str(item)
                    for item in retrieval_output["selected_memory_ids"]
                    if str(item) in allowed_ids
                ][: self.max_selected_memories]
            try:
                loop_output = raw.get("loop") if isinstance(raw.get("loop"), dict) else {}
                progress_output = (
                    loop_output.get("progress")
                    if isinstance(loop_output.get("progress"), dict)
                    else {}
                )
                blocked_subgoals = {
                    _canonical_subgoal(item)
                    for item in progress_output.get("blocked_subgoals") or []
                    if _canonical_subgoal(item)
                }
                next_subgoal = _canonical_subgoal(loop_output.get("next_loop_subgoal"))
                if (
                    str(raw.get("task_status") or "").upper() == "SWITCH_LOOP"
                    and next_subgoal in blocked_subgoals
                ):
                    raise ValueError("next_loop_subgoal reopens a blocked subgoal without new evidence")
                self._validate_contract(raw, state, allowed_ids, current_phase)
                break
            except ValueError as exc:
                last_error = str(exc)
        else:
            raise ValueError(f"controller contract invalid after retry: {last_error}")
        loop = raw.get("loop") if isinstance(raw.get("loop"), dict) else {}
        delta = raw.get("state_delta") if isinstance(raw.get("state_delta"), dict) else {}
        retrieval = raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
        requested = retrieval.get("selected_memory_ids")
        selected = []
        if isinstance(requested, list):
            selected = [str(item) for item in requested if str(item) in allowed_ids][
                : self.max_selected_memories
            ]
        try:
            confidence = max(0.0, min(1.0, float(loop.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        return UnifiedControlDecision(
            task_status=str(raw.get("task_status") or "CONTINUE").upper(),
            research_phase=str(raw.get("research_phase") or current_phase).upper(),
            switch_loop=bool(loop.get("switch", False)),
            reason=str(loop.get("reason") or ""),
            confidence=confidence,
            current_loop_subgoal=str(loop.get("current_loop_subgoal") or ""),
            next_loop_subgoal=str(loop.get("next_loop_subgoal") or ""),
            loop_outcome=str(loop.get("outcome") or "IN_PROGRESS").upper(),
            boundary_basis=str(loop.get("boundary_basis") or "NONE").upper(),
            loop_progress=_normalize_loop_progress(loop.get("progress"), loop_progress),
            state_delta=delta or {"summary": "", "operations": []},
            retrieval_query=str(retrieval.get("query") or ""),
            selected_memory_ids=selected,
            validation_retries=1 if last_error else 0,
            raw_response=raw,
        )

    @staticmethod
    def _validate_contract(
        raw: dict[str, Any],
        state: GlobalIntentState,
        allowed_ids: set[str],
        current_phase: str,
    ) -> None:
        if not isinstance(raw, dict):
            raise ValueError("response must be an object")
        if set(raw) != {"task_status", "research_phase", "loop", "state_delta", "retrieval"}:
            raise ValueError(
                "response must contain exactly task_status, research_phase, loop, state_delta, and retrieval"
            )
        task_status = str(raw.get("task_status") or "").upper()
        if task_status not in TASK_STATUSES:
            raise ValueError("task_status must be CONTINUE, SWITCH_LOOP, or READY_TO_ANSWER")
        research_phase = str(raw.get("research_phase") or "").upper()
        if research_phase not in RESEARCH_PHASES:
            raise ValueError("invalid research_phase")
        loop = raw.get("loop")
        delta = raw.get("state_delta")
        retrieval = raw.get("retrieval")
        if not all(isinstance(value, dict) for value in (loop, delta, retrieval)):
            raise ValueError("loop, state_delta, and retrieval must be objects")
        if not isinstance(loop.get("switch"), bool):
            raise ValueError("loop.switch must be boolean")
        if (task_status == "SWITCH_LOOP") != loop["switch"]:
            raise ValueError("SWITCH_LOOP requires loop.switch=true and other statuses require false")
        outcome = str(loop.get("outcome") or "").upper()
        boundary_basis = str(loop.get("boundary_basis") or "").upper()
        if outcome not in LOOP_OUTCOMES:
            raise ValueError("invalid loop.outcome")
        if boundary_basis not in BOUNDARY_BASES:
            raise ValueError("invalid loop.boundary_basis")
        current_subgoal = str(loop.get("current_loop_subgoal") or "").strip()
        next_subgoal = str(loop.get("next_loop_subgoal") or "").strip()
        if not current_subgoal:
            raise ValueError("loop.current_loop_subgoal cannot be empty")
        if task_status == "CONTINUE" and (outcome != "IN_PROGRESS" or boundary_basis != "NONE"):
            raise ValueError("CONTINUE requires outcome=IN_PROGRESS and boundary_basis=NONE")
        if task_status == "SWITCH_LOOP":
            if outcome == "IN_PROGRESS" or boundary_basis == "NONE":
                raise ValueError("SWITCH_LOOP requires a terminal outcome and non-NONE boundary_basis")
            if not next_subgoal:
                raise ValueError("SWITCH_LOOP requires a non-empty next_loop_subgoal")
        if task_status == "READY_TO_ANSWER" and research_phase != "ANSWER_SYNTHESIS":
            raise ValueError("READY_TO_ANSWER requires ANSWER_SYNTHESIS")
        if research_phase == "ANSWER_SYNTHESIS" and task_status != "READY_TO_ANSWER":
            raise ValueError("ANSWER_SYNTHESIS requires READY_TO_ANSWER")
        if task_status == "READY_TO_ANSWER" and (
            outcome != "RESOLVED" or boundary_basis != "TASK_COMPLETE" or next_subgoal
        ):
            raise ValueError(
                "READY_TO_ANSWER requires RESOLVED, TASK_COMPLETE, and empty next_loop_subgoal"
            )
        progress = loop.get("progress")
        if not isinstance(progress, dict):
            raise ValueError("loop.progress must be an object")
        if task_status == "READY_TO_ANSWER" and not (
            progress.get("answer_stable") is True
            and progress.get("evidence_sufficient") is True
            and progress.get("expected_information_gain") == "LOW"
        ):
            raise ValueError(
                "READY_TO_ANSWER requires stable answer, sufficient evidence, and LOW information gain"
            )
        if task_status != "READY_TO_ANSWER":
            direction = progress.get("research_direction")
            if not isinstance(direction, dict):
                raise ValueError("loop.progress.research_direction must be an object")
            required_direction_fields = ("objective", "stop_condition")
            if any(not str(direction.get(name) or "").strip() for name in required_direction_fields):
                raise ValueError(
                    "research_direction requires objective and stop_condition"
                )
        operations = delta.get("operations")
        if not isinstance(operations, list):
            raise ValueError("state_delta.operations must be a list")
        mode = str(delta.get("mode") or "").upper()
        if not loop["switch"] and (mode != "NOOP" or operations):
            raise ValueError("switch=false requires mode=NOOP and no operations")
        if mode not in {"APPLY", "NOOP"}:
            raise ValueError("state_delta.mode must be APPLY or NOOP")
        if mode == "NOOP" and operations:
            raise ValueError("NOOP cannot contain operations")
        if task_status == "SWITCH_LOOP" and (mode != "APPLY" or not operations):
            raise ValueError("SWITCH_LOOP requires APPLY with at least one durable operation")
        add_targets = {
            "active_subgoals", "confirmed_constraints", "soft_preferences", "candidate_options",
            "rejected_options", "working_hypotheses", "resolved_findings", "open_questions",
            "uncertainties", "next_actions",
        }
        valid_open_ids = {item.item_id for item in state.open_questions}
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError("each operation must be an object")
            operation = str(op.get("operation") or "").upper()
            target = str(op.get("target") or "")
            if operation == "ADD" and target not in add_targets:
                raise ValueError(f"ADD cannot target {target}")
            if operation == "UPDATE" and target != "current_goal":
                raise ValueError("UPDATE may target current_goal only")
            if operation == "RESOLVE" and target != "open_questions":
                raise ValueError("RESOLVE may target open_questions only")
            if operation not in {"ADD", "UPDATE", "RESOLVE"}:
                raise ValueError(f"unsupported operation {operation}")
            if not str(op.get("reason") or "").strip():
                raise ValueError("every operation requires a reason")
            if operation in {"ADD", "UPDATE"} and op.get("value") in (None, ""):
                raise ValueError("ADD/UPDATE requires a value")
            if operation in {"ADD", "UPDATE"} and len(str(op.get("value"))) > 800:
                raise ValueError("StateDelta values must be at most 800 characters")
            evidence_ids = op.get("evidence_ids", [])
            if evidence_ids:
                raise ValueError("controller evidence_ids must be empty")
            target_ids = {str(item) for item in op.get("target_item_ids", [])}
            if operation == "RESOLVE" and target_ids and not target_ids <= valid_open_ids:
                raise ValueError("RESOLVE contains unknown open_question IDs")
            item = op.get("item")
            if operation in {"ADD", "UPDATE"}:
                if item is not None and not isinstance(item, dict):
                    raise ValueError("item must be an object")
                if isinstance(item, dict):
                    status = str(item.get("status") or "tentative")
                    source_type = str(item.get("source_type") or "agent")
                    if status not in VALID_ITEM_STATUSES:
                        raise ValueError(f"invalid item.status {status}")
                    if source_type not in VALID_SOURCE_TYPES:
                        raise ValueError(f"invalid item.source_type {source_type}")
                    try:
                        confidence = float(item.get("confidence", 0.5))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("item.confidence must be numeric") from exc
                    if not 0.0 <= confidence <= 1.0:
                        raise ValueError("item.confidence must be between 0 and 1")
                    for field_name in ("contradicts", "supersedes"):
                        if not isinstance(item.get(field_name, []), list):
                            raise ValueError(f"item.{field_name} must be a list")
        selected = retrieval.get("selected_memory_ids", [])
        if not isinstance(selected, list) or any(str(item) not in allowed_ids for item in selected):
            raise ValueError("retrieval contains unknown memory IDs")


__all__ = ["UnifiedControlDecision", "UnifiedMemoryController", "empty_loop_progress"]
