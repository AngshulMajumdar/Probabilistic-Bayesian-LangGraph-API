from __future__ import annotations
from typing import Callable, Dict, List
import math
from b_langgraph.inference.smc import Particle, SMCConfig, smc_run
from b_langgraph.model.priors import ToolReliabilityState
from b_langgraph.runtime.agent import AgentConfig, BayesianAgent
from b_langgraph.runtime.interfaces import Action, Observation, Tool
from b_langgraph.tools.mock_tools import AskClarificationTool, ConsistencyCheckTool, FastSearchTool, GeoDisambiguateTool, NoisyWebSearchTool, NoticeCheckTool, OfficialNoticeTool, SchoolSearchIN, SchoolSearchUS, VerifiedSearchTool

def _selector_by_reliability(default_scores: Dict[str, float]):
    def selector(query: str, p: Particle, actions: List[Action], t: int) -> Action:
        best, best_score = actions[0], -1.0
        for act in actions:
            score = p.rel.get(act.tool_name).mean() * default_scores.get(act.tool_name, 0.5)
            if score > best_score:
                best, best_score = act, score
        return best
    return selector

def build_stale_vs_verified():
    tools: Dict[str, Tool] = {'fast_search': FastSearchTool(), 'verified_search': VerifiedSearchTool(), 'consistency_check': ConsistencyCheckTool()}
    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            return [Action('verified_search', {'query': query}), Action('fast_search', {'query': query})]
        if t == 1:
            last_ans = particles[0].observations[-1].output.get('answer', '') if particles and particles[0].observations else ''
            return [Action('consistency_check', {'text': last_ans}), Action('verified_search', {'query': query})]
        if t == 2 and particles and particles[0].observations:
            last = particles[0].observations[-1]
            if last.action.tool_name == 'consistency_check' and not bool(last.output.get('ok', True)):
                return [Action('verified_search', {'query': query})]
        return []
    def scorer(query: str, p: Particle, a: Action, obs: Observation, t: int):
        if not obs.ok:
            return 1e-6, False, None
        if a.tool_name in {'fast_search', 'verified_search'}:
            return float(obs.output.get('confidence', 0.1)), (True if obs.output.get('is_verified') else None), None
        if a.tool_name == 'consistency_check':
            ok = bool(obs.output.get('ok', False))
            extra = [(p.actions[-1].tool_name, ok, 1.0)] if p.actions else None
            return (0.95 if ok else 0.05), True, extra
        return 0.2, None, None
    selector = _selector_by_reliability({'verified_search': 0.90, 'fast_search': 0.65, 'consistency_check': 0.80})
    cfg = AgentConfig(smc=SMCConfig(n_particles=24, max_steps=3, max_global_actions=2, seed=7))
    return BayesianAgent(tools=tools, proposer=proposer, scorer=scorer, selector=selector, cfg=cfg)

def build_ambiguous_location():
    tools: Dict[str, Tool] = {'geo_disambiguate': GeoDisambiguateTool(), 'ask_clarification': AskClarificationTool(), 'school_search_us': SchoolSearchUS(), 'school_search_india': SchoolSearchIN()}
    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            return [Action('geo_disambiguate', {'query': query})]
        if t == 1:
            return [Action('ask_clarification', {'hint': query})]
        if t == 2:
            choice = particles[0].observations[-1].output.get('user_choice')
            return [Action('school_search_india', {})] if choice == 'saltlake_kolkata' else [Action('school_search_us', {})]
        return []
    def scorer(query: str, p: Particle, a: Action, obs: Observation, t: int):
        return (0.9 if obs.ok else 1e-6), None, None
    cfg = AgentConfig(smc=SMCConfig(n_particles=16, max_steps=3, max_global_actions=1, seed=11))
    return BayesianAgent(tools=tools, proposer=proposer, scorer=scorer, selector=lambda q, p, acts, t: acts[0], cfg=cfg)

def build_web_vs_official():
    tools: Dict[str, Tool] = {'noisy_web_search': NoisyWebSearchTool(), 'official_notice': OfficialNoticeTool(), 'notice_check': NoticeCheckTool()}
    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            return [Action('official_notice', {'query': query}), Action('noisy_web_search', {'query': query})]
        if t == 1:
            last_ans = particles[0].observations[-1].output.get('answer', '') if particles and particles[0].observations else ''
            return [Action('notice_check', {'text': last_ans}), Action('official_notice', {'query': query})]
        if t == 2 and particles and particles[0].observations:
            last = particles[0].observations[-1]
            if last.action.tool_name == 'notice_check' and not bool(last.output.get('ok', True)):
                return [Action('official_notice', {'query': query})]
        return []
    def scorer(query: str, p: Particle, a: Action, obs: Observation, t: int):
        if not obs.ok:
            return 1e-6, False, None
        if a.tool_name in {'noisy_web_search', 'official_notice'}:
            return float(obs.output.get('confidence', 0.1)), (True if obs.output.get('is_verified') else None), None
        if a.tool_name == 'notice_check':
            ok = bool(obs.output.get('ok', False))
            extra = [(p.actions[-1].tool_name, ok, 1.0)] if p.actions else None
            return (0.97 if ok else 0.03), True, extra
        return 0.2, None, None
    selector = _selector_by_reliability({'official_notice': 0.95, 'noisy_web_search': 0.55, 'notice_check': 0.85})
    cfg = AgentConfig(smc=SMCConfig(n_particles=20, max_steps=3, max_global_actions=2, seed=13))
    return BayesianAgent(tools=tools, proposer=proposer, scorer=scorer, selector=selector, cfg=cfg)

def build_session_learning():
    tools: Dict[str, Tool] = {'fast_search': FastSearchTool(), 'verified_search': VerifiedSearchTool(), 'consistency_check': ConsistencyCheckTool()}
    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            return [Action('fast_search', {'query': query}), Action('verified_search', {'query': query})]
        if t == 1:
            ans = particles[0].observations[-1].output.get('answer', '') if particles and particles[0].observations else ''
            return [Action('consistency_check', {'text': ans}), Action('verified_search', {'query': query})]
        if t == 2 and particles and particles[0].observations:
            last = particles[0].observations[-1]
            if last.action.tool_name == 'consistency_check' and not bool(last.output.get('ok', True)):
                return [Action('verified_search', {'query': query})]
        return []
    def scorer(query: str, p: Particle, a: Action, obs: Observation, t: int):
        if not obs.ok:
            return 1e-6, False, None
        if a.tool_name in {'fast_search', 'verified_search'}:
            return float(obs.output.get('confidence', 0.1)), (True if obs.output.get('is_verified') else None), None
        if a.tool_name == 'consistency_check':
            ok = bool(obs.output.get('ok', False))
            extra = [(p.actions[-1].tool_name, ok, 1.0)] if p.actions else None
            return (0.95 if ok else 0.05), True, extra
        return 0.2, None, None
    score_selector = _selector_by_reliability({'verified_search': 0.90, 'fast_search': 0.65, 'consistency_check': 0.85})
    def run_once(query: str, session_rel: ToolReliabilityState, explore_first: bool):
        def selector(query: str, p: Particle, actions: List[Action], t: int) -> Action:
            if explore_first and t == 0:
                for a in actions:
                    if a.tool_name == 'fast_search':
                        return a
            return score_selector(query, p, actions, t)
        res = smc_run(query=query, tools=tools, proposer=proposer, scorer=scorer, selector=selector, cfg=SMCConfig(n_particles=24, max_steps=3, max_global_actions=2, seed=7), init_rel=session_rel)
        ws = [math.exp(w) for w in res.norm_logw]
        best = res.particles[max(range(len(ws)), key=lambda i: ws[i])]
        answer = next((str(o.output['answer']) for o in reversed(best.observations) if o.output.get('answer')), '')
        return best, answer
    def runner(query: str):
        session_rel = ToolReliabilityState()
        first, first_answer = run_once(query, session_rel, True)
        second, second_answer = run_once(query, first.rel, False)
        return {'run1_answer': first_answer, 'run1_tools': [a.tool_name for a in first.actions], 'run1_tool_means': first.rel.snapshot_means(), 'run2_answer': second_answer, 'run2_tools': [a.tool_name for a in second.actions], 'run2_tool_means': second.rel.snapshot_means()}
    return runner
SCENARIOS: Dict[str, Callable] = {'stale_vs_verified': build_stale_vs_verified, 'ambiguous_location': build_ambiguous_location, 'web_vs_official': build_web_vs_official, 'session_learning': build_session_learning}
