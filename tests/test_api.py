from fastapi.testclient import TestClient

from bayesian_prob_langgraph_api.api.app import app

client = TestClient(app)


def test_health() -> None:
    r = client.get('/api/v1/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_info() -> None:
    r = client.get('/api/v1/info')
    assert r.status_code == 200
    data = r.json()
    assert data['name'] == 'bayesian-probabilistic-langgraph-api'
    assert 'version' in data


def test_tools_and_scenarios() -> None:
    assert client.get('/api/v1/tools').status_code == 200
    scenarios = client.get('/api/v1/scenarios').json()['scenarios']
    assert 'stale_vs_verified' in scenarios
    assert 'session_learning' in scenarios


def test_stale_vs_verified_run() -> None:
    r = client.post(
        '/api/v1/agents/run',
        json={
            'scenario': 'stale_vs_verified',
            'query': 'What is the best time to visit Serbia? Consider weather and budget.',
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert 'Late April to June' in d['answer']
    assert len(d['steps']) >= 1


def test_ambiguous_location_run() -> None:
    r = client.post(
        '/api/v1/agents/run',
        json={'scenario': 'ambiguous_location', 'query': 'Show me the best schools near Salt Lake.'},
    )
    assert r.status_code == 200
    assert any('Kolkata' in str(step['output']) for step in r.json()['steps'])


def test_session_learning_run() -> None:
    r = client.post(
        '/api/v1/agents/run',
        json={
            'scenario': 'session_learning',
            'query': 'What is the best time to visit Serbia? Consider weather and budget.',
        },
    )
    assert r.status_code == 200
    assert 'verified_search' in r.json()['run2_tools']


def test_benchmark_endpoint() -> None:
    r = client.post(
        '/api/v1/benchmarks/run',
        json={
            'scenario': 'web_vs_official',
            'query': 'What is the scholarship deadline?',
            'trials': 5,
        },
    )
    assert r.status_code == 200
    assert r.json()['success_rate'] == 1.0


def test_unknown_scenario_run_returns_404() -> None:
    r = client.post(
        '/api/v1/agents/run',
        json={'scenario': 'not_a_real_scenario', 'query': 'test'},
    )
    assert r.status_code == 404


def test_unknown_scenario_benchmark_returns_404() -> None:
    r = client.post(
        '/api/v1/benchmarks/run',
        json={'scenario': 'not_a_real_scenario', 'query': 'test', 'trials': 3},
    )
    assert r.status_code == 404


def test_benchmark_rejects_non_positive_trials() -> None:
    r = client.post(
        '/api/v1/benchmarks/run',
        json={'scenario': 'web_vs_official', 'query': 'What is the scholarship deadline?', 'trials': 0},
    )
    assert r.status_code == 422
