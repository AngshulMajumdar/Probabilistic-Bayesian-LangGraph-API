# Bayesian Probabilistic LangGraph API

A lightweight, installable FastAPI package for Bayesian and probabilistic LangGraph-style agent orchestration.

This repository exposes a reusable API layer over a small graph-style orchestration runtime with scenario-driven demos for:
- stale versus verified source selection
- session learning after early mistakes
- ambiguous entity or location resolution
- standalone normal/pathological LangGraph-vs-Bayesian-LangGraph demo scripts
- noisy web versus official data selection

It is designed as a clean software artifact for packaging, testing, smoke validation, and extension by other developers.

## Features

- `src/`-based installable package layout
- FastAPI application with documented endpoints
- scenario registry for repeatable demo workflows
- unit and API tests with `pytest`
- benchmark endpoint for repeated runs
- Docker support
- GitHub Actions CI workflow

## Installation

### Standard install

```bash
pip install .
```

### Editable install with development dependencies

```bash
pip install -e .[dev]
```

## Run the API locally

```bash
uvicorn bayesian_prob_langgraph_api.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API docs will be available at:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Main endpoints

### Health check

```bash
curl http://127.0.0.1:8000/api/v1/health
```

### Package metadata

```bash
curl http://127.0.0.1:8000/api/v1/info
```

### Available tools

```bash
curl http://127.0.0.1:8000/api/v1/tools
```

### Available scenarios

```bash
curl http://127.0.0.1:8000/api/v1/scenarios
```

### Run a scenario

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "stale_vs_verified",
    "query": "What is the best time to visit Serbia? Consider weather and budget."
  }'
```

### Run a repeated benchmark

```bash
curl -X POST http://127.0.0.1:8000/api/v1/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "web_vs_official",
    "query": "What is the scholarship deadline?",
    "trials": 20
  }'
```

## Python usage

```python
from b_langgraph.scenarios.registry import build_stale_vs_verified

agent = build_stale_vs_verified()
answer, trace, posterior = agent.run(
    "What is the best time to visit Serbia? Consider weather and budget."
)

print(answer)
print(len(trace.steps))
print(posterior["n_particles"])
```

## Example scripts

- `examples/smoke_test.py` checks the API endpoints locally.
- `examples/python_usage.py` shows direct library usage without HTTP.
- `examples/benchmark_demo.py` runs repeated benchmark requests through the API.
- `examples/normal_rag_demo.py` compares regular LangGraph against Bayesian LangGraph on a simple retrieval case.
- `examples/pathological_rag_demo.py` shows a retry-and-fail regular LangGraph path against a corrective Bayesian path.
- `examples/statistical_benchmark.py` runs the 100-trial comparison benchmark used for the SoftwareX tables.

## Testing

Run the full test suite:

```bash
pytest -q
```

Run the smoke test script:

```bash
python examples/smoke_test.py
```

## Docker

Build the image:

```bash
docker build -t bayesian-prob-langgraph-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 bayesian-prob-langgraph-api
```

## Package structure

```text
src/
  b_langgraph/
    inference/
    model/
    runtime/
    scenarios/
    tools/
  bayesian_prob_langgraph_api/
    api/
examples/
tests/
.github/workflows/
```

## Dependencies

Runtime dependencies:
- `fastapi>=0.115,<1.0`
- `uvicorn>=0.30,<1.0`
- `pydantic>=2.7,<3.0`

Development dependencies:
- `pytest>=8.0,<9.0`
- `httpx>=0.27,<1.0`

## Notes for developers

This package is intentionally lightweight. The included tools are mock tools for demonstrating orchestration logic. They can be replaced with real tools by extending the `Tool` protocol and registering new scenarios.

## License

MIT


## SoftwareX benchmark summary

Using `examples/statistical_benchmark.py`, the following 100-trial results were obtained for the four controlled orchestration scenarios:

| Scenario | Trials | Regular LangGraph Success | Bayesian LangGraph Success | Regular Avg. Steps | Bayesian Avg. Steps |
|---|---:|---:|---:|---:|---:|
| conditional_tool_shift | 100 | 68.0% | 100.0% | 1.00 | 2.00 |
| session_learning | 100 | 84.0% | 100.0% | 2.00 | 2.00 |
| stale_vs_verified | 100 | 39.0% | 100.0% | 1.00 | 2.00 |
| web_vs_official | 100 | 68.0% | 100.0% | 1.00 | 2.00 |
| overall | 400 | 64.8% | 100.0% | 1.25 | 2.00 |
