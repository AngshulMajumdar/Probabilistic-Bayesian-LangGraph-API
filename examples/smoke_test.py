from fastapi.testclient import TestClient
from bayesian_prob_langgraph_api.api.app import app
client = TestClient(app)
print(client.get('/api/v1/health').json())
print(client.get('/api/v1/info').json())
print(client.get('/api/v1/tools').json())
print(client.get('/api/v1/scenarios').json())
print(client.post('/api/v1/agents/run', json={'scenario': 'stale_vs_verified', 'query': 'What is the best time to visit Serbia? Consider weather and budget.'}).json())
