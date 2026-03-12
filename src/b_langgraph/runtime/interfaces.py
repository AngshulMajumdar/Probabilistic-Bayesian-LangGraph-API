from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import json, time
JsonDict = Dict[str, Any]
@runtime_checkable
class Tool(Protocol):
    name: str
    def invoke(self, inp: JsonDict) -> JsonDict: ...
@dataclass(frozen=True)
class Action:
    tool_name: str
    args: JsonDict = field(default_factory=dict)
    def key(self) -> str:
        return f"{self.tool_name}::{json.dumps(self.args, sort_keys=True, ensure_ascii=False)}"
@dataclass(frozen=True)
class Observation:
    action: Action
    output: JsonDict
    ok: bool = True
    error: Optional[str] = None
    latency_s: float = 0.0
    ts: float = field(default_factory=lambda: time.time())
    def short(self, max_chars: int = 240) -> str:
        s = json.dumps(self.output, ensure_ascii=False, sort_keys=True)
        return s if len(s) <= max_chars else s[: max_chars - 3] + '...'
@dataclass
class StepRecord:
    t: int
    proposed_actions: List[Action]
    chosen_action: Action
    observation: Observation
    info: JsonDict = field(default_factory=dict)
@dataclass
class EpisodeTrace:
    query: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    meta: JsonDict = field(default_factory=dict)
    def add_step(self, rec: StepRecord) -> None:
        self.steps.append(rec)
