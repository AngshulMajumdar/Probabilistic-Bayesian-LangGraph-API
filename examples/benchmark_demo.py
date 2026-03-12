from fastapi.testclient import TestClient
from bayesian_prob_langgraph_api.api.app import app


def main() -> None:
    client = TestClient(app)
    response = client.post(
        '/api/v1/benchmarks/run',
        json={
            'scenario': 'web_vs_official',
            'query': 'What is the scholarship deadline?',
            'trials': 10,
        },
    )
    print(response.json())


if __name__ == '__main__':
    main()
