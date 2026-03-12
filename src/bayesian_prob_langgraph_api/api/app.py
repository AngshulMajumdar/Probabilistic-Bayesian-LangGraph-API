from __future__ import annotations
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from bayesian_prob_langgraph_api import __version__
from b_langgraph.scenarios.registry import SCENARIOS
app = FastAPI(title='Bayesian Probabilistic LangGraph API', version=__version__, description='FastAPI wrapper for Bayesian and probabilistic LangGraph-style orchestration demos.')
class RunRequest(BaseModel):
    scenario: str = Field(...)
    query: str = Field(...)
class BenchmarkRequest(BaseModel):
    scenario: str
    query: str
    trials: int = 20

    @field_validator('trials')
    @classmethod
    def validate_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError('trials must be positive')
        return value
@app.get('/api/v1/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}
@app.get('/api/v1/info')
def info() -> Dict[str, Any]:
    return {'name': 'bayesian-probabilistic-langgraph-api', 'version': __version__, 'description': 'Bayesian and probabilistic graph-style agent orchestration API'}
@app.get('/api/v1/tools')
def tools() -> Dict[str, List[str]]:
    return {'tools': ['fast_search', 'verified_search', 'consistency_check', 'geo_disambiguate', 'ask_clarification', 'school_search_us', 'school_search_india', 'noisy_web_search', 'official_notice', 'notice_check']}
@app.get('/api/v1/scenarios')
def scenarios() -> Dict[str, List[str]]:
    return {'scenarios': list(SCENARIOS.keys())}
@app.post('/api/v1/agents/run')
def run_agent(req: RunRequest) -> Dict[str, Any]:
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f'Unknown scenario: {req.scenario}')
    obj = SCENARIOS[req.scenario]()
    if req.scenario == 'session_learning':
        return {'scenario': req.scenario, **obj(req.query)}
    answer, trace, posterior = obj.run(req.query)
    return {'scenario': req.scenario, 'answer': answer, 'steps': [{'t': s.t, 'tool': s.chosen_action.tool_name, 'output': s.observation.output} for s in trace.steps], 'posterior': posterior}
@app.post('/api/v1/benchmarks/run')
def run_benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f'Unknown scenario: {req.scenario}')
    successes = 0
    total_steps = 0
    answers = []
    for _ in range(req.trials):
        obj = SCENARIOS[req.scenario]()
        if req.scenario == 'session_learning':
            out = obj(req.query)
            answer, step_count = out['run2_answer'], len(out['run2_tools'])
        else:
            answer, trace, _ = obj.run(req.query)
            step_count = len(trace.steps)
        success = ('Late April to June' in answer) or ('Kolkata' in answer) or ('April 30' in answer)
        successes += int(success)
        total_steps += step_count
        answers.append(answer)
    return {'scenario': req.scenario, 'trials': req.trials, 'success_rate': successes / max(1, req.trials), 'avg_steps': total_steps / max(1, req.trials), 'sample_answers': answers[:3]}
