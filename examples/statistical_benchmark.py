"""100-run benchmark: regular LangGraph vs Bayesian LangGraph-style orchestration."""
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import pandas as pd
import random
import math


def run_stale_vs_verified(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    query = "What is the current HR portal?"
    gold = "PeopleHub"
    stale_answer = "PeopleHub" if rng.random() < 0.35 else "WorkSphere"

    def fast_search(q: str) -> Dict[str, Any]:
        return {"tool": "fast_search", "answer": stale_answer, "confidence": 0.88, "verified": False, "source_reliability": 0.45}

    def verified_search(q: str) -> Dict[str, Any]:
        return {"tool": "verified_search", "answer": "PeopleHub", "confidence": 0.97, "verified": True, "source_reliability": 0.98}

    class S(TypedDict):
        query: str
        result: Dict[str, Any]
        trace: List[Dict[str, Any]]

    def start(state: S) -> S:
        state["result"] = {}
        state["trace"] = []
        return state

    def node_fast(state: S) -> S:
        out = fast_search(state["query"])
        state["trace"].append(out)
        state["result"] = out
        return state

    def route_after_fast(state: S) -> str:
        return "stop" if state["result"].get("answer") else "fallback"

    def node_verified(state: S) -> S:
        out = verified_search(state["query"])
        state["trace"].append(out)
        state["result"] = out
        return state

    builder = StateGraph(S)
    builder.add_node("start", start)
    builder.add_node("fast", node_fast)
    builder.add_node("verified", node_verified)
    builder.set_entry_point("start")
    builder.add_edge("start", "fast")
    builder.add_conditional_edges("fast", route_after_fast, {"stop": END, "fallback": "verified"})
    builder.add_edge("verified", END)
    graph = builder.compile()

    regular = graph.invoke({"query": query, "result": {}, "trace": []})
    regular_answer = regular["result"]["answer"]
    regular_success = regular_answer == gold
    regular_steps = len(regular["trace"])

    fast = fast_search(query)
    verified = verified_search(query)
    fast_score = 2.0 * fast["confidence"] + 2.0 * fast["source_reliability"] + (1.5 if fast["verified"] else -1.5)
    verified_score = 2.0 * verified["confidence"] + 2.0 * verified["source_reliability"] + (1.5 if verified["verified"] else -1.5)
    chosen = verified if verified_score >= fast_score else fast
    return {"scenario": "stale_vs_verified", "regular_success": regular_success, "bayesian_success": chosen["answer"] == gold, "regular_steps": regular_steps, "bayesian_steps": 2}


def run_session_learning(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    query = "What is the current HR portal?"
    gold = "PeopleHub"
    first_fast_correct = rng.random() < 0.80

    def fast_search_first(q: str) -> Dict[str, Any]:
        return {"tool": "fast_search", "answer": "PeopleHub" if first_fast_correct else "WorkSphere", "confidence": 0.87 if first_fast_correct else 0.86, "verified": False, "source_reliability": 0.60 if first_fast_correct else 0.55}

    def fast_search_retry(q: str) -> Dict[str, Any]:
        return {"tool": "fast_search_retry", "answer": "PeopleHub" if first_fast_correct else "WorkSphere", "confidence": 0.88 if first_fast_correct else 0.85, "verified": False, "source_reliability": 0.60 if first_fast_correct else 0.55}

    def verified_search(q: str) -> Dict[str, Any]:
        return {"tool": "verified_search", "answer": "PeopleHub", "confidence": 0.97, "verified": True, "source_reliability": 0.98}

    class S(TypedDict):
        query: str
        result: Dict[str, Any]
        trace: List[Dict[str, Any]]
        retries: int

    def start(state: S) -> S:
        state["result"] = {}
        state["trace"] = []
        state["retries"] = 0
        return state

    def node_fast(state: S) -> S:
        out = fast_search_first(state["query"]) if state["retries"] == 0 else fast_search_retry(state["query"])
        state["trace"].append(out)
        state["result"] = out
        return state

    def route_after_fast(state: S) -> str:
        return "retry" if state["retries"] == 0 and state["result"]["confidence"] >= 0.84 else "stop"

    def increment_retry(state: S) -> S:
        state["retries"] += 1
        state["trace"].append({"tool": "retry_counter", "count": state["retries"]})
        return state

    builder = StateGraph(S)
    builder.add_node("start", start)
    builder.add_node("fast", node_fast)
    builder.add_node("inc", increment_retry)
    builder.set_entry_point("start")
    builder.add_edge("start", "fast")
    builder.add_conditional_edges("fast", route_after_fast, {"retry": "inc", "stop": END})
    builder.add_edge("inc", "fast")
    graph = builder.compile()

    regular = graph.invoke({"query": query, "result": {}, "trace": [], "retries": 0})
    regular_answer = regular["result"]["answer"]
    regular_success = regular_answer == gold
    regular_steps = len([x for x in regular["trace"] if isinstance(x, dict) and x.get("tool") in ["fast_search", "fast_search_retry"]])

    fast1 = fast_search_first(query)
    fast2 = fast_search_retry(query)
    verified = verified_search(query)
    fast_best = fast1 if fast1["confidence"] >= fast2["confidence"] else fast2
    fast_score = 2.0 * fast_best["confidence"] + 1.5 * fast_best["source_reliability"] + (1.5 if fast_best["verified"] else -1.5)
    verified_score = 2.0 * verified["confidence"] + 1.5 * verified["source_reliability"] + (1.5 if verified["verified"] else -1.5)
    chosen = verified if verified_score >= fast_score else fast_best
    return {"scenario": "session_learning", "regular_success": regular_success, "bayesian_success": chosen["answer"] == gold, "regular_steps": regular_steps, "bayesian_steps": 2}


def run_conditional_tool_shift(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    query = "What is the latest officially notified deadline for portal migration?"
    gold = "April 30"
    default_tool_correct = rng.random() < 0.55

    def fast_search(q: str) -> Dict[str, Any]:
        return {"tool": "fast_search", "answer": "April 30" if default_tool_correct else "March 31", "confidence": 0.89, "verified": False, "source_reliability": 0.55}

    def official_notice(q: str) -> Dict[str, Any]:
        return {"tool": "official_notice", "answer": "April 30", "confidence": 0.98, "verified": True, "source_reliability": 0.99}

    class S(TypedDict):
        query: str
        result: Dict[str, Any]
        trace: List[Dict[str, Any]]

    def start(state: S) -> S:
        state["result"] = {}
        state["trace"] = []
        return state

    def node_fast(state: S) -> S:
        out = fast_search(state["query"])
        state["trace"].append(out)
        state["result"] = out
        return state

    builder = StateGraph(S)
    builder.add_node("start", start)
    builder.add_node("fast", node_fast)
    builder.set_entry_point("start")
    builder.add_edge("start", "fast")
    builder.add_edge("fast", END)
    graph = builder.compile()

    regular = graph.invoke({"query": query, "result": {}, "trace": []})
    regular_answer = regular["result"]["answer"]
    regular_success = regular_answer == gold
    regular_steps = len(regular["trace"])

    fast = fast_search(query)
    official = official_notice(query)
    trigger_bonus = 1.0 if "officially notified" in query.lower() else 0.0
    fast_score = 2.0 * fast["confidence"] + 1.5 * fast["source_reliability"] + (1.5 if fast["verified"] else -1.5)
    official_score = 2.0 * official["confidence"] + 1.5 * official["source_reliability"] + (1.5 if official["verified"] else -1.5) + trigger_bonus
    chosen = official if official_score >= fast_score else fast
    return {"scenario": "conditional_tool_shift", "regular_success": regular_success, "bayesian_success": chosen["answer"] == gold, "regular_steps": regular_steps, "bayesian_steps": 2}


def run_web_vs_official(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    query = "What is the official portal migration deadline?"
    gold = "April 30"
    web_correct = rng.random() < 0.60

    def noisy_web_search(q: str) -> Dict[str, Any]:
        return {"tool": "noisy_web_search", "answer": "April 30" if web_correct else "March 31", "confidence": 0.84, "verified": False, "source_reliability": 0.58}

    def official_db(q: str) -> Dict[str, Any]:
        return {"tool": "official_db", "answer": "April 30", "confidence": 0.99, "verified": True, "source_reliability": 0.99}

    class S(TypedDict):
        query: str
        result: Dict[str, Any]
        trace: List[Dict[str, Any]]

    def start(state: S) -> S:
        state["result"] = {}
        state["trace"] = []
        return state

    def node_web(state: S) -> S:
        out = noisy_web_search(state["query"])
        state["trace"].append(out)
        state["result"] = out
        return state

    builder = StateGraph(S)
    builder.add_node("start", start)
    builder.add_node("web", node_web)
    builder.set_entry_point("start")
    builder.add_edge("start", "web")
    builder.add_edge("web", END)
    graph = builder.compile()

    regular = graph.invoke({"query": query, "result": {}, "trace": []})
    regular_answer = regular["result"]["answer"]
    regular_success = regular_answer == gold
    regular_steps = len(regular["trace"])

    web = noisy_web_search(query)
    official = official_db(query)
    web_score = 2.0 * web["confidence"] + 1.4 * web["source_reliability"] + (1.5 if web["verified"] else -1.5)
    official_score = 2.0 * official["confidence"] + 1.4 * official["source_reliability"] + (1.5 if official["verified"] else -1.5)
    chosen = official if official_score >= web_score else web
    return {"scenario": "web_vs_official", "regular_success": regular_success, "bayesian_success": chosen["answer"] == gold, "regular_steps": regular_steps, "bayesian_steps": 2}


def main() -> None:
    all_rows = []
    for i in range(100):
        seed = 1000 + i
        all_rows.append(run_stale_vs_verified(seed))
        all_rows.append(run_session_learning(seed))
        all_rows.append(run_conditional_tool_shift(seed))
        all_rows.append(run_web_vs_official(seed))
    raw_df = pd.DataFrame(all_rows)
    summary = raw_df.groupby("scenario").agg(trials=("scenario", "count"), regular_success_rate=("regular_success", "mean"), bayesian_success_rate=("bayesian_success", "mean"), regular_avg_steps=("regular_steps", "mean"), bayesian_avg_steps=("bayesian_steps", "mean")).reset_index()
    summary["regular_success_rate"] = (100 * summary["regular_success_rate"]).round(1)
    summary["bayesian_success_rate"] = (100 * summary["bayesian_success_rate"]).round(1)
    summary["regular_avg_steps"] = summary["regular_avg_steps"].round(2)
    summary["bayesian_avg_steps"] = summary["bayesian_avg_steps"].round(2)
    overall = pd.DataFrame([{
        "scenario": "overall",
        "trials": len(raw_df),
        "regular_success_rate": round(100 * raw_df["regular_success"].mean(), 1),
        "bayesian_success_rate": round(100 * raw_df["bayesian_success"].mean(), 1),
        "regular_avg_steps": round(raw_df["regular_steps"].mean(), 2),
        "bayesian_avg_steps": round(raw_df["bayesian_steps"].mean(), 2),
    }])
    final_table = pd.concat([summary, overall], ignore_index=True)
    print("\nFINAL TABLE\n")
    print(final_table.to_string(index=False))
    raw_df.to_csv("langgraph_regular_vs_bayesian_raw.csv", index=False)
    final_table.to_csv("langgraph_regular_vs_bayesian_summary.csv", index=False)
    print("\nSaved:")
    print(" - langgraph_regular_vs_bayesian_raw.csv")
    print(" - langgraph_regular_vs_bayesian_summary.csv")


if __name__ == "__main__":
    main()
