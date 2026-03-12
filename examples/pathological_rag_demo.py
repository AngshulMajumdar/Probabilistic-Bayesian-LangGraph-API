"""Pathological RAG demo: regular LangGraph retries and fails while Bayesian LangGraph-style corrects and succeeds."""
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import math
import json

stale_docs = [{
    "id": "ST-001",
    "title": "Old HR Portal Migration Note",
    "text": "The HR portal is WorkSphere. Employees should use WorkSphere for leave requests and payslips.",
    "answer": "WorkSphere",
    "verified": False,
    "source_reliability": 0.42,
    "retrieval_score": 0.93,
}]
verified_docs = [{
    "id": "VF-001",
    "title": "Current Portal Access Policy",
    "text": "The current HR portal is PeopleHub. Employees must use PeopleHub for leave requests, payslips, and profile updates.",
    "answer": "PeopleHub",
    "verified": True,
    "source_reliability": 0.98,
    "retrieval_score": 0.74,
}]

gold_answer = "PeopleHub"
query = "What is the current HR portal?"


def stale_retriever(query: str) -> Dict[str, Any]:
    doc = stale_docs[0]
    return {
        "tool": "stale_retriever",
        "doc_id": doc["id"],
        "title": doc["title"],
        "context": doc["text"],
        "answer": doc["answer"],
        "verified": doc["verified"],
        "source_reliability": doc["source_reliability"],
        "retrieval_score": doc["retrieval_score"],
        "notes": "Fast cached retrieval from stale internal source.",
    }


def verified_retriever(query: str) -> Dict[str, Any]:
    doc = verified_docs[0]
    return {
        "tool": "verified_retriever",
        "doc_id": doc["id"],
        "title": doc["title"],
        "context": doc["text"],
        "answer": doc["answer"],
        "verified": doc["verified"],
        "source_reliability": doc["source_reliability"],
        "retrieval_score": doc["retrieval_score"],
        "notes": "Slower authoritative retrieval from official policy source.",
    }


def generate_from_context(retrieval_out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_answer": retrieval_out["answer"],
        "confidence": 0.90 if retrieval_out["retrieval_score"] > 0.85 else 0.96,
        "based_on_doc": retrieval_out["doc_id"],
    }


def validator(answer: str) -> Dict[str, Any]:
    plausible = answer in ["WorkSphere", "PeopleHub"]
    return {"plausible": plausible, "needs_retry": plausible, "notes": "Answer format plausible; retrying retrieval for confirmation."}


class RegularRAGState(TypedDict):
    query: str
    retrieval: Dict[str, Any]
    generation: Dict[str, Any]
    validation: Dict[str, Any]
    trace: List[Dict[str, Any]]
    retries: int
    max_retries: int


def rg_start(state: RegularRAGState) -> RegularRAGState:
    state["retrieval"] = {}
    state["generation"] = {}
    state["validation"] = {}
    state["trace"] = []
    state["retries"] = 0
    if "max_retries" not in state:
        state["max_retries"] = 2
    return state


def rg_retrieve_stale(state: RegularRAGState) -> RegularRAGState:
    out = stale_retriever(state["query"])
    state["retrieval"] = out
    state["trace"].append({"step": "retrieve_stale", "output": out})
    return state


def rg_generate(state: RegularRAGState) -> RegularRAGState:
    out = generate_from_context(state["retrieval"])
    state["generation"] = out
    state["trace"].append({"step": "generate", "output": out})
    return state


def rg_validate(state: RegularRAGState) -> RegularRAGState:
    out = validator(state["generation"]["generated_answer"])
    state["validation"] = out
    state["trace"].append({"step": "validate", "output": out})
    return state


def rg_increment_retry(state: RegularRAGState) -> RegularRAGState:
    state["retries"] += 1
    state["trace"].append({"step": "retry_increment", "output": {"retries": state["retries"]}})
    return state


def rg_route_after_validate(state: RegularRAGState) -> str:
    if state["validation"]["needs_retry"] and state["retries"] < state["max_retries"]:
        return "retry_same_path"
    return "stop"


regular_builder = StateGraph(RegularRAGState)
regular_builder.add_node("start", rg_start)
regular_builder.add_node("retrieve_stale", rg_retrieve_stale)
regular_builder.add_node("generate", rg_generate)
regular_builder.add_node("validate", rg_validate)
regular_builder.add_node("increment_retry", rg_increment_retry)
regular_builder.set_entry_point("start")
regular_builder.add_edge("start", "retrieve_stale")
regular_builder.add_edge("retrieve_stale", "generate")
regular_builder.add_edge("generate", "validate")
regular_builder.add_conditional_edges("validate", rg_route_after_validate, {"retry_same_path": "increment_retry", "stop": END})
regular_builder.add_edge("increment_retry", "retrieve_stale")
regular_graph = regular_builder.compile()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def bayesian_rag(query: str) -> Dict[str, Any]:
    stale_hit = stale_retriever(query)
    verified_hit = verified_retriever(query)
    stale_gen = generate_from_context(stale_hit)
    verified_gen = generate_from_context(verified_hit)
    stale_score = 1.2 * stale_hit["retrieval_score"] + 1.4 * stale_hit["source_reliability"] + (1.5 if stale_hit["verified"] else -1.5) + 0.4 * stale_gen["confidence"]
    verified_score = 1.2 * verified_hit["retrieval_score"] + 1.4 * verified_hit["source_reliability"] + (1.5 if verified_hit["verified"] else -1.5) + 0.4 * verified_gen["confidence"]
    posterior_verified = sigmoid(verified_score - stale_score)
    chosen_retrieval = verified_hit if verified_score >= stale_score else stale_hit
    chosen_generation = verified_gen if verified_score >= stale_score else stale_gen
    return {
        "query": query,
        "trace": [
            {"step": "retrieve_stale", "output": stale_hit},
            {"step": "retrieve_verified", "output": verified_hit},
            {"step": "generate_stale", "output": stale_gen},
            {"step": "generate_verified", "output": verified_gen},
        ],
        "chosen_retrieval": chosen_retrieval,
        "chosen_generation": chosen_generation,
        "stale_score": round(stale_score, 4),
        "verified_score": round(verified_score, 4),
        "posterior_verified": round(posterior_verified, 6),
        "steps": 2,
    }


def main() -> None:
    regular_result = regular_graph.invoke({
        "query": query,
        "retrieval": {},
        "generation": {},
        "validation": {},
        "trace": [],
        "retries": 0,
        "max_retries": 2,
    })
    bayesian_result = bayesian_rag(query)
    regular_answer = regular_result["generation"]["generated_answer"]
    bayesian_answer = bayesian_result["chosen_generation"]["generated_answer"]
    summary = {
        "query": query,
        "gold_answer": gold_answer,
        "regular_langgraph_answer": regular_answer,
        "regular_langgraph_success": regular_answer == gold_answer,
        "regular_langgraph_retrieval_calls": len([x for x in regular_result["trace"] if x["step"] == "retrieve_stale"]),
        "regular_langgraph_retries": regular_result["retries"],
        "bayesian_langgraph_answer": bayesian_answer,
        "bayesian_langgraph_success": bayesian_answer == gold_answer,
        "bayesian_langgraph_steps": bayesian_result["steps"],
        "bayesian_posterior_verified": bayesian_result["posterior_verified"],
    }
    print("=" * 100)
    print("QUERY")
    print("=" * 100)
    print(query)
    print("\n" + "=" * 100)
    print("REGULAR LANGGRAPH PATHOLOGICAL RAG RESULT")
    print("=" * 100)
    print(json.dumps(regular_result, indent=2))
    print("\n" + "=" * 100)
    print("BAYESIAN LANGGRAPH PATHOLOGICAL RAG RESULT")
    print("=" * 100)
    print(json.dumps(bayesian_result, indent=2))
    print("\n" + "=" * 100)
    print("COMPACT SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
