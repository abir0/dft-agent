import json
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from planner_graph import generate_plan
from executor import execute_node # Using the executor we designed

class AgentState(TypedDict):
    request: str
    history: List[Dict[str, Any]]
    plan: Dict[str, Any]
    execution_results: Dict[str, Any]
    error: str | None # To handle potential errors


def planner_node(state: AgentState) -> dict:
    """
    A wrapper for your generate_plan function to make it a compatible graph node.
    """
    print("--- Running Planner Node ---")
    try:
        result = generate_plan(
            request_text=state["request"],
            history=state["history"]
            # We can add hints and code here if needed
        )
        return {"plan": result.get("plan")}
    except Exception as e:
        return {"error": f"Planner failed: {e}"}

def executor_node(state: AgentState) -> dict:
    """
    This is the executor node we designed. It's already in the correct format.
    """
    print("--- Running Executor Node ---")
    try:
        results = execute_node(state)
        return results
    except Exception as e:
        return {"error": f"Executor failed: {e}"}


# Define the state graph
graph = StateGraph(AgentState)

# Add the planner and executor as nodes
graph.add_node("plan", planner_node)
graph.add_node("execute", executor_node)

# Set the entry point and define the flow
graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

planner_executor_graph = graph.compile(checkpointer=MemorySaver())