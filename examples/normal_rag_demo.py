"""Normal RAG-style demo: regular LangGraph vs Bayesian LangGraph-style orchestration.

This script is self-contained and mirrors the Colab-verified demo used during software validation.
"""
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import math
import json


def fast_search(query: str) -> Dict[str, Any]:
    return {
        "tool": "fast_search",
        "answer": "WorkSphere",
        "confidence": 0.88,
        "verified": False,
        "source_reliability": 0.45,
        "notes": "Cached answer from stale internal search.",
    }


def verified_search(query: str) -> Dict[str, Any]:
    return {
        "tool": "verified_search",
        "answer": "PeopleHub",
        "confidence": 0.97,
        "verified": True,
        "source_reliability": 0.98,
        "notes": "Official current policy portal.",
    }


class RegularState(TypedDict):
    query: str
    result: Dict[str, Any]
    trace: List[Dict[str, Any]]


def regular_start(state: RegularState) -> RegularState:
    state["trace"] = []
    state["result"] = {}
    return state


def regular_fast_node(state: RegularState) -> RegularState:
    out = fast_search(state["query"])
    state["trace"].append(out)
    state["result"] = out
    return state


def regular_should_stop(state: RegularState) -> str:
    return "stop" if state["result"].get("answer") else "fallback"


def regular_verified_node(state: RegularState) -> RegularState:
    out = verified_search(state["query"])
    state["trace"].append(out)
    state["result"] = out
    return state


regular_builder = StateGraph(RegularState)
regular_builder.add_node("start", regular_start)
regular_builder.add_node("fast_search", regular_fast_node)
regular_builder.add_node("verified_search", regular_verified_node)
regular_builder.set_entry_point("start")
regular_builder.add_edge("start", "fast_search")
regular_builder.add_conditional_edges(
    "fast_search",
    regular_should_stop,
    {"stop": END, "fallback": "verified_search"},
)
regular_builder.add_edge("verified_search", END)
regular_graph = regular_builder.compile()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def bayesian_langgraph_run(query: str) -> Dict[str, Any]:
    fast = fast_search(query)
    verified = verified_search(query)
    fast_score = 2.0 * fast["confidence"] + 2.0 * fast["source_reliability"] + (1.5 if fast["verified"] else -1.5)
    verified_score = 2.0 * verified["confidence"] + 2.0 * verified["source_reliability"] + (1.5 if verified["verified"] else -1.5)
    posterior_verified = sigmoid(verified_score - fast_score)
    chosen = verified if verified_score >= fast_score else fast
    return {
        "query": query,
        "trace": [fast, verified],
        "chosen": chosen,
        "fast_score": round(fast_score, 4),
        "verified_score": round(verified_score, 4),
        "posterior_verified": round(posterior_verified, 6),
        "steps": 2,
    }


def main() -> None:
    query = "What is the current HR portal?"
    regular_result = regular_graph.invoke({"query": query, "result": {}, "trace": []})
    bayesian_result = bayesian_langgraph_run(query)
    regular_answer = regular_result["result"]["answer"]
    bayesian_answer = bayesian_result["chosen"]["answer"]
    gold = "PeopleHub"
    summary = {
        "gold_answer": gold,
        "regular_langgraph_answer": regular_answer,
        "regular_langgraph_success": regular_answer == gold,
        "regular_langgraph_steps": len(regular_result["trace"]),
        "bayesian_langgraph_answer": bayesian_answer,
        "bayesian_langgraph_success": bayesian_answer == gold,
        "bayesian_langgraph_steps": bayesian_result["steps"],
        "posterior_verified": bayesian_result["posterior_verified"],
    }
    print("=" * 90)
    print("QUERY")
    print("=" * 90)
    print(query)
    print("\n" + "=" * 90)
    print("REGULAR LANGGRAPH RESULT")
    print("=" * 90)
    print(json.dumps(regular_result, indent=2))
    print("\n" + "=" * 90)
    print("BAYESIAN LANGGRAPH RESULT")
    print("=" * 90)
    print(json.dumps(bayesian_result, indent=2))
    print("\n" + "=" * 90)
    print("COMPACT SUMMARY")
    print("=" * 90)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
