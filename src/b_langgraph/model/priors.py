from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
@dataclass
class BetaBelief:
    alpha: float = 1.0
    beta: float = 1.0
    def mean(self) -> float:
        d = self.alpha + self.beta
        return self.alpha / d if d > 0 else 0.5
    def update(self, success: bool, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        if success:
            self.alpha += weight
        else:
            self.beta += weight
@dataclass
class ToolReliabilityState:
    beliefs: Dict[str, BetaBelief] = field(default_factory=dict)
    def get(self, tool_name: str) -> BetaBelief:
        if tool_name not in self.beliefs:
            self.beliefs[tool_name] = BetaBelief()
        return self.beliefs[tool_name]
    def update(self, tool_name: str, success: bool, weight: float = 1.0) -> None:
        self.get(tool_name).update(success=success, weight=weight)
    def snapshot_means(self) -> Dict[str, float]:
        return {k: v.mean() for k, v in self.beliefs.items()}
