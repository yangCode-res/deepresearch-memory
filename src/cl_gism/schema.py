"""Versioned, JSON-serializable data structures for the first CL-GISM step.

The first implementation deliberately keeps the schema dependency-free.  It
defines the four memory layers from the design document and the State Delta
record that will connect them in the next implementation step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, ClassVar, Mapping
from uuid import uuid4


class SchemaValidationError(ValueError):
    """Raised when a schema object violates an invariant."""


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SourceType(_StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    PAPER = "paper"
    WEB = "web"
    EXPERIMENT = "experiment"
    SYSTEM = "system"


class MemoryStatus(_StrEnum):
    ACTIVE = "active"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"


class StateItemKind(_StrEnum):
    CURRENT_GOAL = "current_goal"
    ACTIVE_SUBGOAL = "active_subgoal"
    CONFIRMED_CONSTRAINT = "confirmed_constraint"
    SOFT_PREFERENCE = "soft_preference"
    CANDIDATE_OPTION = "candidate_option"
    REJECTED_OPTION = "rejected_option"
    WORKING_HYPOTHESIS = "working_hypothesis"
    RESOLVED_FINDING = "resolved_finding"
    OPEN_QUESTION = "open_question"
    UNCERTAINTY = "uncertainty"
    NEXT_ACTION = "next_action"


class DeltaOperation(_StrEnum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    CONFIRM = "CONFIRM"
    WEAKEN = "WEAKEN"
    REJECT = "REJECT"
    OVERRIDE = "OVERRIDE"
    RESOLVE = "RESOLVE"
    REOPEN = "REOPEN"
    BRANCH = "BRANCH"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Create a stable, human-readable identifier for a schema object."""

    return f"{prefix}_{uuid4().hex}"


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _enum_value(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {k: _enum_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_value(v) for v in value]
    return value


class SchemaModel:
    """Small common API shared by all schema records."""

    _required_prefix: ClassVar[str | None] = None

    def validate(self) -> None:
        """Validate invariants; subclasses extend this method."""

        if self._required_prefix:
            object_id = getattr(self, self._id_field, "")
            if not isinstance(object_id, str) or not object_id.startswith(self._required_prefix + "_"):
                raise SchemaValidationError(
                    f"{self.__class__.__name__}.{self._id_field} must start with "
                    f"'{self._required_prefix}_'"
                )

    @property
    def _id_field(self) -> str:
        return "id"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _enum_value(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class EvidencePointer(SchemaModel):
    """Reference to immutable Raw Memory used to support a higher-level fact."""

    raw_memory_id: str
    locator: str | None = None
    excerpt: str | None = None

    def validate(self) -> None:
        if not self.raw_memory_id.startswith("raw_"):
            raise SchemaValidationError("EvidencePointer.raw_memory_id must reference a raw_* object")


@dataclass
class TaskAnchor(SchemaModel):
    """Stable task intent and success criteria.

    Anchor updates are represented as new versions by the caller; ordinary
    Turn processing must not mutate an existing anchor in place.
    """

    task_id: str
    original_goal: str
    success_criteria: list[str] = field(default_factory=list)
    immutable_constraints: list[str] = field(default_factory=list)
    domain: str | None = None
    version: int = 1
    created_at: str = field(default_factory=utc_now)
    evidence_ids: list[str] = field(default_factory=list)
    supersedes_anchor_id: str | None = None
    anchor_id: str = field(default_factory=lambda: new_id("anchor"))

    _required_prefix: ClassVar[str] = "anchor"

    @property
    def _id_field(self) -> str:
        return "anchor_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("TaskAnchor.task_id cannot be empty")
        if not self.original_goal.strip():
            raise SchemaValidationError("TaskAnchor.original_goal cannot be empty")
        if self.version < 1:
            raise SchemaValidationError("TaskAnchor.version must be >= 1")
        if any(not item.startswith("raw_") for item in self.evidence_ids):
            raise SchemaValidationError("TaskAnchor.evidence_ids must contain raw_* IDs")
        if self.supersedes_anchor_id and not self.supersedes_anchor_id.startswith("anchor_"):
            raise SchemaValidationError("TaskAnchor.supersedes_anchor_id must reference an anchor_* object")


@dataclass
class StateItem(SchemaModel):
    """One typed, versionable fact in Global Intent State."""

    kind: StateItemKind
    value: Any
    status: MemoryStatus = MemoryStatus.TENTATIVE
    confidence: float = 0.5
    source_type: SourceType = SourceType.AGENT
    evidence_ids: list[str] = field(default_factory=list)
    created_in_loop: str | None = None
    updated_in_loop: str | None = None
    valid_time: str | None = None
    contradicts: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    user_confirmed: bool = False
    item_id: str = field(default_factory=lambda: new_id("state"))

    _required_prefix: ClassVar[str] = "state"

    @property
    def _id_field(self) -> str:
        return "item_id"

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaValidationError("StateItem.confidence must be between 0 and 1")
        if any(not item.startswith("raw_") and not item.startswith("loop_") for item in self.evidence_ids):
            raise SchemaValidationError("StateItem.evidence_ids must contain raw_* or loop_* IDs")
        if self.created_in_loop and not self.created_in_loop.startswith("loop_"):
            raise SchemaValidationError("StateItem.created_in_loop must reference a loop_* object")
        if self.updated_in_loop and not self.updated_in_loop.startswith("loop_"):
            raise SchemaValidationError("StateItem.updated_in_loop must reference a loop_* object")
        if any(not item.startswith("state_") for item in self.contradicts + self.supersedes):
            raise SchemaValidationError("StateItem.contradicts/supersedes must contain state_* IDs")
        if self.user_confirmed and self.source_type is not SourceType.USER:
            raise SchemaValidationError("user_confirmed StateItem must have source_type='user'")


@dataclass
class GlobalIntentState(SchemaModel):
    """Current task state; all state facts are retained as typed StateItems."""

    task_id: str
    state_version: int = 1
    current_goal: StateItem | None = None
    active_subgoals: list[StateItem] = field(default_factory=list)
    confirmed_constraints: list[StateItem] = field(default_factory=list)
    soft_preferences: list[StateItem] = field(default_factory=list)
    candidate_options: list[StateItem] = field(default_factory=list)
    rejected_options: list[StateItem] = field(default_factory=list)
    working_hypotheses: list[StateItem] = field(default_factory=list)
    resolved_findings: list[StateItem] = field(default_factory=list)
    open_questions: list[StateItem] = field(default_factory=list)
    uncertainties: list[StateItem] = field(default_factory=list)
    next_actions: list[StateItem] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)
    source_loop_id: str | None = None
    state_id: str = field(default_factory=lambda: new_id("global_state"))

    _required_prefix: ClassVar[str] = "global_state"

    @property
    def _id_field(self) -> str:
        return "state_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("GlobalIntentState.task_id cannot be empty")
        if self.state_version < 1:
            raise SchemaValidationError("GlobalIntentState.state_version must be >= 1")
        if self.source_loop_id and not self.source_loop_id.startswith("loop_"):
            raise SchemaValidationError("GlobalIntentState.source_loop_id must reference a loop_* object")
        groups = {
            "current_goal": self.current_goal,
            "active_subgoals": self.active_subgoals,
            "confirmed_constraints": self.confirmed_constraints,
            "soft_preferences": self.soft_preferences,
            "candidate_options": self.candidate_options,
            "rejected_options": self.rejected_options,
            "working_hypotheses": self.working_hypotheses,
            "resolved_findings": self.resolved_findings,
            "open_questions": self.open_questions,
            "uncertainties": self.uncertainties,
            "next_actions": self.next_actions,
        }
        expected = {
            "current_goal": StateItemKind.CURRENT_GOAL,
            "active_subgoals": StateItemKind.ACTIVE_SUBGOAL,
            "confirmed_constraints": StateItemKind.CONFIRMED_CONSTRAINT,
            "soft_preferences": StateItemKind.SOFT_PREFERENCE,
            "candidate_options": StateItemKind.CANDIDATE_OPTION,
            "rejected_options": StateItemKind.REJECTED_OPTION,
            "working_hypotheses": StateItemKind.WORKING_HYPOTHESIS,
            "resolved_findings": StateItemKind.RESOLVED_FINDING,
            "open_questions": StateItemKind.OPEN_QUESTION,
            "uncertainties": StateItemKind.UNCERTAINTY,
            "next_actions": StateItemKind.NEXT_ACTION,
        }
        for field_name, value in groups.items():
            items = [] if value is None else [value] if isinstance(value, StateItem) else value
            if not isinstance(items, list):
                raise SchemaValidationError(f"GlobalIntentState.{field_name} must contain StateItem records")
            for item in items:
                if not isinstance(item, StateItem):
                    raise SchemaValidationError(f"GlobalIntentState.{field_name} contains a non-StateItem")
                item.validate()
                if item.kind is not expected[field_name]:
                    raise SchemaValidationError(
                        f"StateItem {item.item_id} has kind={item.kind.value}, expected {expected[field_name].value}"
                    )

    def all_items(self) -> list[StateItem]:
        groups: list[StateItem] = []
        if self.current_goal:
            groups.append(self.current_goal)
        for name in (
            "active_subgoals",
            "confirmed_constraints",
            "soft_preferences",
            "candidate_options",
            "rejected_options",
            "working_hypotheses",
            "resolved_findings",
            "open_questions",
            "uncertainties",
            "next_actions",
        ):
            groups.extend(getattr(self, name))
        return groups


@dataclass
class LoopMemory(SchemaModel):
    """A completed local task unit spanning one or more Turns/actions."""

    task_id: str
    subgoal: str
    context: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    user_feedback: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str | None = None
    state_delta_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    loop_id: str = field(default_factory=lambda: new_id("loop"))

    _required_prefix: ClassVar[str] = "loop"

    @property
    def _id_field(self) -> str:
        return "loop_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("LoopMemory.task_id cannot be empty")
        if not self.subgoal.strip():
            raise SchemaValidationError("LoopMemory.subgoal cannot be empty")
        if self.status in (MemoryStatus.RESOLVED, MemoryStatus.SUPERSEDED) and not self.ended_at:
            raise SchemaValidationError("Completed LoopMemory must have ended_at")
        if any(not item.startswith("raw_") for item in self.evidence_ids):
            raise SchemaValidationError("LoopMemory.evidence_ids must contain raw_* IDs")
        if any(not item.startswith("delta_") for item in self.state_delta_ids):
            raise SchemaValidationError("LoopMemory.state_delta_ids must contain delta_* IDs")


@dataclass(frozen=True)
class RawMemory(SchemaModel):
    """Source event or document retained as ground-truth evidence.

    Attribute replacement is blocked after creation.  Callers should also
    treat nested ``content`` and ``metadata`` values as immutable payloads.
    """

    task_id: str
    source_type: SourceType
    content: Any
    content_type: str = "text/plain"
    occurred_at: str = field(default_factory=utc_now)
    recorded_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_loop_id: str | None = None
    raw_id: str = field(default_factory=lambda: new_id("raw"))

    _required_prefix: ClassVar[str] = "raw"

    @property
    def _id_field(self) -> str:
        return "raw_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("RawMemory.task_id cannot be empty")
        if not self.content_type.strip():
            raise SchemaValidationError("RawMemory.content_type cannot be empty")
        if self.parent_loop_id and not self.parent_loop_id.startswith("loop_"):
            raise SchemaValidationError("RawMemory.parent_loop_id must reference a loop_* object")


@dataclass
class StateDeltaOperation(SchemaModel):
    """One auditable mutation to Global Intent State."""

    operation: DeltaOperation
    target: str
    reason: str
    evidence_ids: list[str] = field(default_factory=list)
    value: Any = None
    old_value: Any = None
    loop_id: str | None = None
    operation_id: str = field(default_factory=lambda: new_id("delta_op"))

    _required_prefix: ClassVar[str] = "delta_op"

    @property
    def _id_field(self) -> str:
        return "operation_id"

    def validate(self) -> None:
        super().validate()
        if not self.target.strip():
            raise SchemaValidationError("StateDeltaOperation.target cannot be empty")
        if not self.reason.strip():
            raise SchemaValidationError("StateDeltaOperation.reason cannot be empty")
        if any(not item.startswith("raw_") and not item.startswith("loop_") for item in self.evidence_ids):
            raise SchemaValidationError("StateDeltaOperation.evidence_ids must contain raw_* or loop_* IDs")
        if self.loop_id and not self.loop_id.startswith("loop_"):
            raise SchemaValidationError("StateDeltaOperation.loop_id must reference a loop_* object")


@dataclass
class StateDelta(SchemaModel):
    """Versioned batch of State Delta operations generated from one Loop."""

    task_id: str
    from_state_version: int
    to_state_version: int
    operations: list[StateDeltaOperation] = field(default_factory=list)
    generated_from_loop_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    delta_id: str = field(default_factory=lambda: new_id("delta"))

    _required_prefix: ClassVar[str] = "delta"

    @property
    def _id_field(self) -> str:
        return "delta_id"

    def validate(self) -> None:
        super().validate()
        if not self.task_id:
            raise SchemaValidationError("StateDelta.task_id cannot be empty")
        if self.from_state_version < 1 or self.to_state_version <= self.from_state_version:
            raise SchemaValidationError("StateDelta versions must advance monotonically")
        if self.generated_from_loop_id and not self.generated_from_loop_id.startswith("loop_"):
            raise SchemaValidationError("StateDelta.generated_from_loop_id must reference a loop_* object")
        for operation in self.operations:
            operation.validate()


__all__ = [
    "DeltaOperation",
    "EvidencePointer",
    "GlobalIntentState",
    "LoopMemory",
    "MemoryStatus",
    "RawMemory",
    "SchemaValidationError",
    "SourceType",
    "StateDelta",
    "StateDeltaOperation",
    "StateItem",
    "StateItemKind",
    "TaskAnchor",
    "new_id",
    "utc_now",
]
