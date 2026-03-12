from b_langgraph.model.priors import ToolReliabilityState
from b_langgraph.runtime.agent import summarize_posterior
from b_langgraph.scenarios.registry import (
    build_ambiguous_location,
    build_stale_vs_verified,
    build_web_vs_official,
)



def test_tool_reliability_updates() -> None:
    rel = ToolReliabilityState()
    assert rel.get('tool_a').mean() == 0.5
    rel.update('tool_a', True)
    rel.update('tool_a', False)
    assert 0.49 < rel.get('tool_a').mean() < 0.76



def test_stale_vs_verified_core() -> None:
    agent = build_stale_vs_verified()
    answer, trace, posterior = agent.run(
        'What is the best time to visit Serbia? Consider weather and budget.'
    )
    assert 'Late April to June' in answer
    assert posterior['n_particles'] > 0
    assert len(trace.steps) >= 1



def test_ambiguous_location_core() -> None:
    agent = build_ambiguous_location()
    answer, trace, posterior = agent.run('Show me the best schools near Salt Lake.')
    assert 'Kolkata' in ' '.join(str(s.observation.output) for s in trace.steps)
    assert posterior['n_particles'] > 0



def test_web_vs_official_core() -> None:
    agent = build_web_vs_official()
    answer, trace, posterior = agent.run('What is the scholarship deadline?')
    assert 'April 30' in answer
    assert posterior['steps_run'] >= 1



def test_posterior_summary_shape() -> None:
    agent = build_stale_vs_verified()
    _, _, posterior = agent.run('What is the best time to visit Serbia? Consider weather and budget.')
    assert 'top_particles' in posterior
    assert isinstance(posterior['top_particles'], list)
    assert posterior['n_particles'] >= 1
