# backend/graph/qe_workflow.py
from __future__ import annotations

import json
import operator
import uuid
from ast import literal_eval
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage

# Local tools
from backend.tools.qe_tools import (
    QEParams,
    JobSpec,
    prepare_qe_inputs,
    run_qe_local,
    run_qe_aiida,
    suggest_qe_retry_params,
)

class DFTState(TypedDict, total=False):
    messages: Annotated[List, operator.add]
    structure_path: str
    pseudo_dir: str
    workdir: str
    kmesh: Tuple[int, int, int]
    params_json: Optional[str]
    run_mode: str  # "local" or "aiida"
    output_path: Optional[str]
    parsed: Optional[Dict]
    stdout_tail: Optional[str]
    retries: int
    max_retries: int
    done: bool
    error: Optional[str]

TOOLS = [prepare_qe_inputs, run_qe_local, run_qe_aiida, suggest_qe_retry_params]
tool_node = ToolNode(TOOLS)

# ---------------- helpers ----------------

def _coerce_payload(x) -> Dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        # try JSON then Python-literal (dict with single quotes)
        try:
            return json.loads(x)
        except Exception:
            try:
                v = literal_eval(x)
                return v if isinstance(v, dict) else {}
            except Exception:
                return {}
    return {}

def _latest_tool_payload(state: DFTState, tool_name: str) -> Dict:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == tool_name:
            return _coerce_payload(m.content)
    return {}

# ---------------- nodes ----------------

def plan_node(state: DFTState) -> DFTState:
    """
    Seed the graph: ask ToolNode to run prepare_qe_inputs.
    ToolNode executes tools in the *last AIMessage's* tool_calls.
    """
    # NOTE: ToolNode requires last message to be AIMessage with tool_calls.
    # Docs: "runs the tools requested in the last AIMessage".
    # https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph_prebuilt.ToolNode.html
    st = dict(state)
    st.setdefault("messages", [])
    st["messages"].append(
        AIMessage(
            content="",
            tool_calls=[{
                "type": "tool_call",
                "id": "prep-"+uuid.uuid4().hex[:6],
                "name": "prepare_qe_inputs",
                "args": {
                    "structure_path": st["structure_path"],
                    "workdir": st["workdir"],
                    "pseudo_dir": st["pseudo_dir"],
                    "kmesh": list(st.get("kmesh", (3, 3, 1))),
                    "params_json": st.get("params_json"),
                },
            }],
        )
    )
    st.setdefault("retries", 0)
    st.setdefault("max_retries", 3)
    return st

def route_after_tools(state: DFTState) -> str:
    """
    Pure router: decide next hop by looking at the last ToolMessage only.
    Do NOT mutate state here (conditional edge functions should be pure).
    """
    for m in reversed(state.get("messages", [])):
        if isinstance(m, ToolMessage):
            if m.name == "prepare_qe_inputs":
                return "submit_node"
            if m.name == "run_qe_local":
                return "check_node"
            if m.name == "run_qe_aiida":
                return "finish_node"
            if m.name == "suggest_qe_retry_params":
                return "prepare_node"
    return "finish_node"

def _params_json_for_retry(state: DFTState) -> str:
    # If user provided params, keep them; otherwise empty dicts are OK (tool will set defaults)
    if state.get("params_json"):
        return state["params_json"]
    return json.dumps({"control": {}, "system": {}, "electrons": {}})

def submit_node(state: DFTState) -> DFTState:
    """
    Build the JobSpec directly from state (do NOT rely on prepare's payload).
    Then queue run_qe_local or run_qe_aiida as a tool call.
    """
    st = dict(state)
    job = JobSpec(
        structure_path=st["structure_path"],
        workdir=st["workdir"],
        pseudo_dir=st["pseudo_dir"],
        kmesh=tuple(st.get("kmesh", (3, 3, 1))),
        params=(QEParams(**json.loads(st["params_json"])) if st.get("params_json") else None),
        code_command="pw.x",
        input_filename="qe.pwi",
        output_filename="qe.pwo",
        run_mode=st.get("run_mode", "local"),
    )

    if job.run_mode == "aiida":
        call = {
            "type": "tool_call",
            "id": "runaiida-"+uuid.uuid4().hex[:6],
            "name": "run_qe_aiida",
            "args": {"job_json": json.dumps(asdict(job))}
        }
    else:
        call = {
            "type": "tool_call",
            "id": "runlocal-"+uuid.uuid4().hex[:6],
            "name": "run_qe_local",
            "args": {"job_json": json.dumps(asdict(job))}
        }

    st["messages"].append(AIMessage(content="", tool_calls=[call]))
    return st

def check_node(state: DFTState) -> DFTState:
    """
    Read the latest run payload, decide if converged; if not, ask for safer params.
    """
    st = dict(state)
    payload = _latest_tool_payload(st, "run_qe_local")
    if payload:
        st["output_path"] = payload.get("output_path")
        st["parsed"] = payload.get("parsed")
        st["stdout_tail"] = payload.get("stdout_tail", "")

    parsed = st.get("parsed") or {}
    if bool(parsed.get("converged")):
        st["done"] = True
        return st

    retries = int(st.get("retries", 0))
    if retries >= int(st.get("max_retries", 3)):
        st["error"] = f"SCF not converged after {retries} retries"
        st["done"] = True
        return st

    st["messages"].append(
        AIMessage(
            content="",
            tool_calls=[{
                "type": "tool_call",
                "id": "suggest-"+uuid.uuid4().hex[:6],
                "name": "suggest_qe_retry_params",
                "args": {
                    "params_json": _params_json_for_retry(st),
                    "stdout_tail": st.get("stdout_tail", "") or "",
                },
            }]
        )
    )
    st["retries"] = retries + 1
    return st

def finish_node(state: DFTState) -> DFTState:
    return state

def build_qe_graph() -> StateGraph:
    g = StateGraph(DFTState)
    g.add_node("prepare_node", plan_node)
    g.add_node("tools", tool_node)
    g.add_node("submit_node", submit_node)
    g.add_node("check_node", check_node)
    g.add_node("finish_node", finish_node)

    g.add_edge(START, "prepare_node")
    g.add_edge("prepare_node", "tools")
    g.add_conditional_edges("tools", route_after_tools, {
        "prepare_node": "prepare_node",
        "submit_node": "submit_node",
        "check_node": "check_node",
        "finish_node": "finish_node",
    })
    g.add_edge("submit_node", "tools")
    g.add_edge("check_node", "tools")
    g.add_edge("finish_node", END)
    return g

if __name__ == "__main__":
    """
    Minimal CLI:
    python -m backend.graph.qe_workflow \\
        --structure_path data/CO_on_Pt111.traj \\
        --pseudo_dir pseudos/PBE \\
        --workdir runs/demo \\
        --run_mode local
    """
    import argparse
    from langgraph.checkpoint.memory import MemorySaver

    parser = argparse.ArgumentParser()
    parser.add_argument("--structure_path", required=True)
    parser.add_argument("--pseudo_dir", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--kmesh", nargs=3, type=int, default=[3, 3, 1])
    parser.add_argument("--run_mode", choices=["local", "aiida"], default="local")
    parser.add_argument("--thread_id", default=None, help="LangGraph thread id for checkpointing")
    args = parser.parse_args()

    state: DFTState = {
        "messages": [],
        "structure_path": args.structure_path,
        "pseudo_dir": args.pseudo_dir,
        "workdir": args.workdir,
        "kmesh": tuple(args.kmesh),
        "params_json": None,
        "run_mode": args.run_mode,
        "retries": 0,
        "max_retries": 3,
    }

    memory = MemorySaver()
    graph = build_qe_graph().compile(checkpointer=memory)
    thread_id = args.thread_id or (Path(args.workdir).name + "-" + uuid.uuid4().hex[:8])
    cfg = {"configurable": {"thread_id": thread_id}}
    final = graph.invoke(state, config=cfg)
    print(json.dumps({k: v for k, v in final.items() if k in ("parsed", "error", "done", "output_path")}, indent=2))
