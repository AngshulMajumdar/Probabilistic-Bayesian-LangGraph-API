from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import math, random, time
from b_langgraph.inference.utils import LRUCache, ess_from_logweights, normalize_logweights, systematic_resample
from b_langgraph.model.priors import ToolReliabilityState
from b_langgraph.runtime.interfaces import Action, Observation, Tool
@dataclass
class Particle:
    actions: List[Action] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    rel: ToolReliabilityState = field(default_factory=ToolReliabilityState)
    logw: float = 0.0
    def copy_shallow(self) -> 'Particle':
        return Particle(actions=list(self.actions), observations=list(self.observations), rel=ToolReliabilityState(beliefs={k: type(v)(v.alpha, v.beta) for k, v in self.rel.beliefs.items()}), logw=self.logw)
@dataclass
class SMCConfig:
    n_particles: int = 24
    n_max_particles: int = 64
    max_steps: int = 4
    max_global_actions: int = 3
    ess_ratio_threshold: float = 0.40
    time_budget_s: float = 5.0
    cache_maxsize: int = 256
    seed: Optional[int] = 0
@dataclass
class SMCResult:
    particles: List[Particle]
    norm_logw: List[float]
    steps_run: int
    time_used_s: float
    ess_history: List[float] = field(default_factory=list)
def run_tool(tools: Dict[str, Tool], action: Action, cache: LRUCache) -> Observation:
    key = 'tool:' + action.key()
    cached = cache.get(key)
    if cached is not None:
        return cached
    t0 = time.time()
    try:
        out = tools[action.tool_name].invoke(action.args)
        obs = Observation(action=action, output=out, ok=True, latency_s=time.time() - t0)
    except Exception as e:
        obs = Observation(action=action, output={}, ok=False, error=str(e), latency_s=time.time() - t0)
    cache.set(key, obs)
    return obs
def smc_run(query: str, tools: Dict[str, Tool], proposer: Callable, scorer: Callable, selector: Callable, cfg: Optional[SMCConfig] = None, init_rel: Optional[ToolReliabilityState] = None) -> SMCResult:
    cfg = cfg or SMCConfig()
    rng = random.Random(cfg.seed)
    cache = LRUCache(maxsize=cfg.cache_maxsize)
    start = time.time()
    def clone_rel(r: ToolReliabilityState) -> ToolReliabilityState:
        return ToolReliabilityState(beliefs={k: type(v)(v.alpha, v.beta) for k, v in r.beliefs.items()})
    particles = [Particle(rel=clone_rel(init_rel)) if init_rel else Particle() for _ in range(cfg.n_particles)]
    ess_hist, steps_run = [], 0
    for t in range(cfg.max_steps):
        if time.time() - start >= cfg.time_budget_s:
            break
        global_actions = proposer(query, particles, t)[: cfg.max_global_actions]
        if not global_actions:
            break
        new_particles, logw_list = [], []
        for p in particles:
            p2 = p.copy_shallow()
            a = selector(query, p2, global_actions, t)
            obs = run_tool(tools, a, cache)
            ret = scorer(query, p2, a, obs, t)
            success_prob, label, extra = (ret + (None,))[:3]
            sp = min(max(float(success_prob), 1e-12), 1.0)
            p2.logw += math.log(sp)
            if label is not None:
                p2.rel.update(a.tool_name, bool(label), 1.0)
            if extra:
                for tool_name, succ, wt in extra:
                    p2.rel.update(tool_name, bool(succ), float(wt))
            p2.actions.append(a)
            p2.observations.append(obs)
            new_particles.append(p2)
            logw_list.append(p2.logw)
        particles = new_particles
        steps_run = t + 1
        ess = ess_from_logweights(logw_list)
        ess_hist.append(ess)
        if ess / max(1, len(particles)) < cfg.ess_ratio_threshold:
            idxs = systematic_resample(logw_list, rng)
            particles = [particles[i].copy_shallow() for i in idxs]
            for pp in particles:
                pp.logw = 0.0
    final_logw = [p.logw for p in particles] if particles else []
    return SMCResult(particles=particles, norm_logw=normalize_logweights(final_logw) if final_logw else [], steps_run=steps_run, time_used_s=time.time() - start, ess_history=ess_hist)
