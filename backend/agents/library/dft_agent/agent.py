"""Simplified DFT Agent Graph

This module defines a streamlined DFT agent similar in structure to the provided
Data Analyst agent example. The agent:
 - Chats about its DFT capabilities
 - Plans multi-step workflows (explicit planning loop in model instructions)
 - Executes registered DFT tools (with automatic thread / workspace injection)
 - Summarizes / analyzes results after tool executions
 - Handles tool errors, attempts automatic fixes (up to 3 retries), and reports
   failures with guidance (leveraging an `error_handler` tool when appropriate)

State retains lightweight workflow context (working directory, structures, etc.).
"""

import asyncio
import inspect
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition

from backend.agents.library.dft_agent.tool_registry import TOOL_REGISTRY
from backend.agents.llm import get_model, settings
from backend.agents.tools import python_repl
from backend.utils.workspace import async_get_workspace_path


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    Simple DFT Agent State that tracks only essential workflow context:
    - Current working directory (user-specific workspace)
    - Thread ID for workspace management
    - Active structures being worked on
    - Last calculation results
    - Current workflow step

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    # Current working directory for all DFT files (user-specific workspace)
    working_directory: str

    # Thread/chat ID for workspace management
    thread_id: str

    # Current structures being worked on (max 3-5 to keep it simple)
    current_structures: Dict[str, str]  # {name: file_path}

    # Last calculation results for reference
    last_calculation: Optional[Dict[str, Any]]

    # Current workflow context (what we're trying to accomplish)
    current_workflow: Optional[str]  # e.g., "adsorption_energy", "surface_energy"

    # Simple step counter for workflows
    workflow_step: int


#############################################
# Tools
#############################################

# Collect all registered DFT tools plus generic python REPL
dft_tools = [*TOOL_REGISTRY.values(), python_repl]


async def initialize_agent_state(config: RunnableConfig) -> Dict[str, Any]:
    """Initialize default DFT agent state with workspace support (non-blocking)."""
    thread_id = config.get("configurable", {}).get("thread_id")
    workspace_path = await async_get_workspace_path(thread_id)

    return {
        "working_directory": str(workspace_path),
        "thread_id": thread_id,
        "current_structures": {},
        "last_calculation": None,
        "current_workflow": None,
        "workflow_step": 0,
    }


def update_structure_registry(state: AgentState, name: str, file_path: str) -> AgentState:
    """Add or update a structure in the registry (keep only last 5)."""
    current_structures = state.get("current_structures", {})
    current_structures[name] = file_path

    # Keep only last 5 structures to prevent state bloat
    if len(current_structures) > 5:
        # Remove oldest entries (this is a simple approach)
        items = list(current_structures.items())
        current_structures = dict(items[-5:])

    return {**state, "current_structures": current_structures}


def update_calculation_result(state: AgentState, result: Dict[str, Any]) -> AgentState:
    """Update the last calculation result."""
    return {**state, "last_calculation": result}


def start_workflow(state: AgentState, workflow_type: str) -> AgentState:
    """Start a new workflow."""
    return {**state, "current_workflow": workflow_type, "workflow_step": 1}


def next_workflow_step(state: AgentState) -> AgentState:
    """Advance to the next workflow step."""
    current_step = state.get("workflow_step", 0)
    return {**state, "workflow_step": current_step + 1}


def reset_workflow(state: AgentState) -> AgentState:
    """Reset workflow state."""
    return {**state, "current_workflow": None, "workflow_step": 0}


#############################################
# System Instructions
#############################################

current_date = datetime.now().strftime("%B %d, %Y")

instructions = f"""
You are an expert Density Functional Theory (DFT) autonomous research assistant.
Today's date is {current_date}.

PRIMARY OBJECTIVE: Understand the user's materials science goal, design a concise step-by-step DFT workflow, execute necessary tool calls, then return a clear scientific summary (methods, key parameters, results, next steps).

CAPABILITIES / TOOL GROUPS:
 1. Structure generation/manipulation: generate_bulk, create_supercell, generate_slab, add_adsorbate, add_vacuum
 2. Setup & execution: generate_qe_input, submit_local_job, check_job_status, read_output_file, extract_energy
 3. Optimization & analysis: geometry_optimization, relax_bulk, relax_slab, generate_kpoint_mesh, get_kpath_bandstructure
 4. Databases & info: search_materials_project, analyze_crystal_structure, find_pseudopotentials
 5. Convergence: kpoint_convergence_test, cutoff_convergence_test, slab_thickness_convergence, vacuum_convergence_test
 6. Properties: calculate_adsorption_energy, calculate_surface_energy, calculate_formation_energy, validate_calculation
 7. Data management: create_calculations_database, store_calculation, query_calculations, export_results, search_similar_calculations
 8. Troubleshooting: error_handler
 9. Lightweight data parsing / quick calculations / custom post-processing: python_repl (use for inspecting tool output files, computing derived quantities, small pandas dataframes, or generating simple plots if needed)

STATE (use sparingly / reference, do NOT dump raw): working_directory, current_structures (<=5), last_calculation, current_workflow, workflow_step.

WORKSPACE: All user-specific files live under {settings.ROOT_PATH}/WORKSPACE/<thread_id>/ with subfolders (structures/, calculations/, results/, convergence/, db/).
Always use the working_directory from state for all file operations.
If thread_id is available, provide it when calling tools to ensure correct workspace usage.
Provide absolute path for any file arguments.

PROCESS (follow unless user explicitly wants ad-hoc answer):
 Step 0: Clarify goal (if ambiguous) otherwise restate it concisely.
 Step 1: Draft a minimal numbered PLAN (1..N) mapping actions to tools (only those likely needed). Include rationale (short phrases). If purely conceptual question, skip tools and answer directly.
 Step 2: Execute plan step-by-step via tool calls. Only call one logically atomic tool per step (unless trivial pairing). Avoid redundant calculations. Use python_repl ONLY for lightweight analysis, parsing generated files (e.g., extracting specific numbers, computing differences), summarizing results, or generating simple helper scripts—NOT for long-running ab initio tasks.
 Step 3: After each tool execution, briefly interpret its output and decide next step; stop early if goal satisfied.
 Step 4: Produce FINAL SUMMARY: goal, methods (key parameters: k-points, cutoffs, pseudopotentials), main results (energies etc.), validation notes, and recommended next steps or improvements.

ERROR RECOVERY:
 - If a tool error occurs: analyze likely cause, adjust parameters, retry (<=3 total attempts per failing logical step).
 - Use error_handler tool if cause unclear after first failure.
 - After exhausting retries: report failure reason + suggested manual fix.
 - If DFT calculation fails check the CRASH file first for clues. (use python_repl to read/extract if needed).

STYLE & CONSTRAINTS:
 - Do NOT expose raw tool JSON directly; interpret scientifically.
 - When using python_repl, keep code concise, comment briefly, and avoid repeated large file reads; prefer targeted extraction.
 - Never write outside the workspace and avoid overwriting important input/output files unless intentional & stated.
 - Explain DFT concepts briefly when helpful (concise, no fluff).
 - Ensure physical reasonableness (e.g., k-point density, vacuum thickness, cutoff ranges).
 - Prefer convergence tests before production property calculations unless user declines.
 - If prior results in state are reusable, reference them instead of recalculating.
 - Always justify key parameter choices (e.g., k-point grid heuristic).

If user only greets or asks basic theory: respond directly (no tools, but mention capabilities if relevant). If they request a workflow/action: create plan then use tools.
"""


#############################################
# Model Wrapper
#############################################


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(dft_tools)
    preprocessor = RunnableLambda(
        lambda state: [
            SystemMessage(
                content=instructions.replace(
                    "<thread_id>", state.get("thread_id", "<thread_id>") or "<thread_id>"
                )
            )
        ]
        + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Primary DFT agent reasoning / planning node."""
    # Initialize once per thread: only add defaults if not already present in state.
    init_updates: Dict[str, Any] = {}
    if "thread_id" not in state or "working_directory" not in state:
        init_updates = await initialize_agent_state(config)

    # Compose full state for model invocation (do NOT mutate original in-place for persistence semantics)
    full_state = {**init_updates, **state}

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(full_state, config)

    # Return messages plus any initialization keys so they persist in LangGraph state
    return {**init_updates, "messages": [response]}


#############################################
# Error Handling & Tool Execution
#############################################


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=(
                    "Error encountered: "
                    f"{repr(error)}\nAttempt to diagnose, adjust parameters, and retry. "
                    "If unclear, call error_handler tool. After 3 failed retries, summarize failure."
                ),
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


async def custom_tool_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute tool calls asynchronously with thread/workspace injection and error capture."""
    thread_id = state.get("thread_id")
    if not thread_id:
        thread_id = config.get("configurable", {}).get("thread_id")
    # Don't rely on in-place mutation; include in returned update instead.

    messages = state["messages"]
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        # Still propagate thread_id if we just discovered it
        update: Dict[str, Any] = {"messages": []}
        if thread_id:
            update["thread_id"] = thread_id
        return update

    tool_messages: list[ToolMessage] = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = (tool_call["args"] or {}).copy()
        tool_spec = next((t for t in dft_tools if t.name == tool_name), None)
        underlying_ann = getattr(tool_spec.func, "__annotations__", {}) if tool_spec else {}
        if thread_id and "_thread_id" in underlying_ann:
            tool_args["_thread_id"] = thread_id

        if not tool_spec:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found. Suggest verifying name or updating registry.",
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        try:
            fn = tool_spec.func
            if inspect.iscoroutinefunction(fn):
                result = await fn(**tool_args)  # type: ignore[arg-type]
            else:
                result = await asyncio.to_thread(fn, **tool_args)
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content=f"Execution error in {tool_name}: {e}",
                    tool_call_id=tool_call["id"],
                )
            )

    # Ensure thread_id is persisted in graph state
    update: Dict[str, Any] = {"messages": tool_messages}
    if thread_id:
        update["thread_id"] = thread_id
    return update


def create_tool_node_with_fallback(_tools: list) -> RunnableLambda:
    return RunnableLambda(custom_tool_node).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


#############################################
# Graph Definition
#############################################

workflow = StateGraph(AgentState)

workflow.add_node("dft", acall_model)
workflow.add_node("tools", create_tool_node_with_fallback(dft_tools))

workflow.set_entry_point("dft")
workflow.add_conditional_edges("dft", tools_condition, ["tools", END])
workflow.add_edge("tools", "dft")
workflow.add_edge("dft", END)

# Export compiled graph
dft_agent = workflow.compile()
