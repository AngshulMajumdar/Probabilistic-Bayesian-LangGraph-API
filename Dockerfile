FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install .
EXPOSE 8000
CMD ["uvicorn", "bayesian_prob_langgraph_api.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
