from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
from b_langgraph.inference.smc import Particle, SMCConfig, SMCResult, smc_run
from b_langgraph.runtime.interfaces import Action, EpisodeTrace, StepRecord, Tool
@dataclass
class AgentConfig:
    smc: SMCConfig = field(default_factory=SMCConfig)
    stop_top_mass: float = 0.80
def summarize_posterior(result: SMCResult, top_k: int = 3):
    if not result.particles:
        return {'n_particles': 0, 'top_particles': []}
    ws = [math.exp(w) for w in result.norm_logw]
    order = sorted(range(len(ws)), key=lambda i: ws[i], reverse=True)[:top_k]
    top = []
    for i in order:
        p = result.particles[i]
        top.append({'mass': ws[i], 'actions': [{'tool': a.tool_name, 'args': a.args} for a in p.actions], 'tool_reliability_means': p.rel.snapshot_means()})
    return {'n_particles': len(result.particles), 'steps_run': result.steps_run, 'time_used_s': round(result.time_used_s, 4), 'ess_history': [round(x, 4) for x in result.ess_history], 'top_particles': top}
class BayesianAgent:
    def __init__(self, tools: Dict[str, Tool], proposer: Callable, scorer: Callable, selector: Callable, cfg: Optional[AgentConfig] = None):
        self.tools, self.proposer, self.scorer, self.selector = tools, proposer, scorer, selector
        self.cfg = cfg or AgentConfig()
    def run(self, query: str) -> Tuple[str, EpisodeTrace, dict]:
        trace = EpisodeTrace(query=query)
        res = smc_run(query, self.tools, self.proposer, self.scorer, self.selector, cfg=self.cfg.smc)
        post = summarize_posterior(res)
        if not res.particles:
            trace.final_answer = 'No particles survived.'
            return trace.final_answer, trace, post
        ws = [math.exp(w) for w in res.norm_logw]
        best = res.particles[max(range(len(ws)), key=lambda i: ws[i])]
        for t, (a, obs) in enumerate(zip(best.actions, best.observations)):
            trace.add_step(StepRecord(t=t, proposed_actions=[a], chosen_action=a, observation=obs, info={'tool_rel_means': best.rel.snapshot_means()}))
        final = next((str(obs.output['answer']) for obs in reversed(best.observations) if isinstance(obs.output, dict) and obs.output.get('answer')), None)
        trace.final_answer = final or (best.observations[-1].short() if best.observations else 'No observations.')
        return trace.final_answer, trace, post
