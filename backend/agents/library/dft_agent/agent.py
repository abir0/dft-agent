"""
DFT Agent Implementation

A comprehensive DFT agent that orchestrates computational materials science workflows
using the DFT tools package.
"""

import asyncio
import inspect
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition

from backend.agents.library.dft_agent.tool_registry import TOOL_REGISTRY
from backend.agents.llm import get_model, settings
from backend.utils.workspace import (
    async_get_workspace_path,
    extract_thread_id_from_config,
)


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


# Get all DFT tools from registry
dft_tools = list(TOOL_REGISTRY.values())


async def initialize_agent_state(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize default DFT agent state with workspace support (non-blocking)."""
    thread_id = extract_thread_id_from_config(config)
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


# System message for DFT agent
current_date = datetime.now().strftime("%B %d, %Y")

instructions = f"""
    You are an expert DFT (Density Functional Theory) agent specialized in computational materials science.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE DIRECTLY.

    PLANNING CAPABILITIES:
    - You can create, modify, and execute workflow plans for complex DFT tasks
    - Plans consist of sequential steps using available tools
    - Each plan has a goal and breaks complex workflows into manageable steps
    - You can execute plans step-by-step and track progress

    Your capabilities include:

    1. STRUCTURE GENERATION & MANIPULATION:
       - Generate bulk crystal structures (generate_bulk)
       - Create supercells (create_supercell)
       - Generate surface slabs (generate_slab)
       - Add adsorbates to surfaces (add_adsorbate)
       - Add vacuum layers (add_vacuum)

    2. CALCULATION SETUP & EXECUTION:
       - Generate Quantum ESPRESSO input files (generate_qe_input)
       - Submit and monitor calculations (submit_local_job, check_job_status)
       - Parse calculation outputs (read_output_file, extract_energy)

    3. STRUCTURE OPTIMIZATION & ANALYSIS:
       - Perform geometry optimizations (geometry_optimization)
       - Relax bulk and slab structures (relax_bulk, relax_slab)
       - Generate k-point meshes (generate_kpoint_mesh)
       - Get band structure paths (get_kpath_bandstructure)

    4. MATERIALS DATABASE INTEGRATION:
       - Search Materials Project database (search_materials_project)
       - Analyze crystal structures (analyze_crystal_structure)
       - Find pseudopotentials (find_pseudopotentials)

    5. CONVERGENCE TESTING:
       - Test k-point convergence (kpoint_convergence_test)
       - Test cutoff energy convergence (cutoff_convergence_test)
       - Test slab thickness convergence (slab_thickness_convergence)
       - Test vacuum convergence (vacuum_convergence_test)

    6. PROPERTY CALCULATIONS:
       - Calculate adsorption energies (calculate_adsorption_energy)
       - Calculate surface energies (calculate_surface_energy)
       - Calculate formation energies (calculate_formation_energy)
       - Validate calculations (validate_calculation)

    7. DATABASE MANAGEMENT:
       - Create calculation databases (create_calculations_database)
       - Store and query calculations (store_calculation, query_calculations)
       - Export results (export_results)
       - Search similar calculations (search_similar_calculations)

    8. ERROR HANDLING & TROUBLESHOOTING:
       - Handle calculation errors (error_handler)
       - Provide troubleshooting guidance

    IMPORTANT GUIDELINES:
    - Always break down complex workflows into logical steps
    - Use appropriate convergence testing before production calculations
    - Store important results in databases for future reference
    - Provide clear explanations of DFT concepts and methods
    - Suggest best practices for computational efficiency
    - Handle errors gracefully and provide solutions
    - Check input parameters for physical reasonableness
    - Recommend appropriate k-point densities and cutoffs
    - Consider system size effects and finite-size corrections
    - Validate results against experimental data when available
    - Document calculation parameters for reproducibility
    - All calculation files are organized in user-specific workspaces under: {settings.ROOT_PATH}/WORKSPACE
    - Each user gets their own workspace directory based on their chat session ID
    - Files are automatically organized into subdirectories: structures/, calculations/, results/, etc.
    - Always explain what each tool does and why you're using it
    - Provide step-by-step workflows for complex DFT tasks

    STATE MANAGEMENT:
    You have access to a simple state that tracks:
    - working_directory: Where all DFT files are stored
    - current_structures: Recently created/loaded structures (max 5)
    - last_calculation: Results from the most recent calculation
    - current_workflow: Active workflow type (adsorption_energy, surface_energy, etc.)
    - workflow_step: Current step number in the workflow

    Use this state to:
    - Reference previous calculations and structures
    - Track progress through multi-step workflows
    - Avoid repeating unnecessary calculations
    - Provide context-aware assistance
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(dft_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_chat(state: AgentState, config: RunnableConfig) -> AgentState:
    """Initial chat node for user interaction and workflow routing."""
    # Initialize DFT state if not present
    if "working_directory" not in state or "thread_id" not in state:
        state.update(await initialize_agent_state(config))

    # Simple chat instructions for initial interaction
    chat_instructions = f"""
    You are a friendly DFT (Density Functional Theory) assistant.

    Your role is to:
    1. Greet users and understand their computational materials science needs
    2. Explain DFT concepts in simple terms when asked
    3. Route complex DFT tasks to the specialized DFT agent
    4. Provide quick answers for simple questions
    5. Help users get started with DFT calculations

    If the user asks for:
    - Complex DFT calculations, workflows, or multi-step processes → Use tools
    - Simple explanations, greetings, or general questions → Answer directly
    - Help getting started → Provide guidance and then use tools if needed

    Today's date is {datetime.now().strftime("%B %d, %Y")}.
    """

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    chat_model = m.bind_tools([])  # No tools for initial chat

    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=chat_instructions)] + state["messages"],
        name="ChatModifier",
    )

    chat_runnable = preprocessor | chat_model
    response = await chat_runnable.ainvoke(state, config)

    return {"messages": [response]}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Main DFT agent node for complex calculations and workflows."""
    # Initialize DFT state if not present
    if "working_directory" not in state or "thread_id" not in state:
        state.update(await initialize_agent_state(config))

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def route_conversation(state: AgentState) -> str:
    """Route conversation between chat and DFT agent based on last message."""
    messages = state["messages"]
    if not messages:
        return "chat"

    last_message = messages[-1]

    # Check if the last message is from an AI and has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # If it's the first user message or a simple question, go to chat first
    user_messages = [
        msg for msg in messages if hasattr(msg, "type") and msg.type == "human"
    ]

    if len(user_messages) <= 1:
        return "chat"

    # For subsequent messages, check if they need DFT tools
    last_user_message = user_messages[-1].content.lower() if user_messages else ""

    # Keywords that suggest DFT tool usage
    dft_keywords = [
        "calculate",
        "generate",
        "optimize",
        "relax",
        "convergence",
        "structure",
        "slab",
        "bulk",
        "adsorbate",
        "energy",
        "dft",
        "quantum espresso",
        "vasp",
        "materials project",
        "database",
        "k-point",
        "cutoff",
        "pseudopotential",
        "workflow",
    ]

    # If message contains DFT keywords or is asking for complex tasks, route to DFT agent
    if any(keyword in last_user_message for keyword in dft_keywords):
        return "dft_agent"

    # Otherwise, handle with chat
    return "chat"


# Agent functions
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n Please fix your mistakes and try again. "
                f"If you need help, use the error_handler tool to get troubleshooting guidance.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


async def custom_tool_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute tool calls without blocking the event loop.

    Any synchronous tool function is offloaded to a worker thread via
    asyncio.to_thread. Native async tool functions (if added later) are
    awaited directly.
    """
    thread_id = state.get("thread_id")

    messages = state["messages"]
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_messages: list[ToolMessage] = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = (tool_call["args"] or {}).copy()

        if thread_id and tool_name in TOOL_REGISTRY:
            tool_args["_thread_id"] = thread_id

        tool_func = next((t for t in dft_tools if t.name == tool_name), None)

        if not tool_func:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found",
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        try:
            fn = tool_func.func
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
                    content=f"Error executing {tool_name}: {e}",
                    tool_call_id=tool_call["id"],
                )
            )

    return {"messages": tool_messages}


def create_tool_node_with_fallback(tools: list) -> dict:
    # Return our custom tool node instead of the default ToolNode
    return RunnableLambda(custom_tool_node)


# Define the original DFT agent graph (kept for backward compatibility)
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("chat", acall_chat)
workflow.add_node("dft_agent", acall_model)
workflow.add_node("tools", custom_tool_node)

# Define edges
workflow.set_entry_point("chat")

# From chat, route to DFT agent or stay in chat
workflow.add_conditional_edges(
    "chat",
    route_conversation,
    {
        "chat": "__end__",  # End conversation if handled by chat
        "dft_agent": "dft_agent",  # Route to DFT agent for complex tasks
    },
)

# From DFT agent, check for tool calls
workflow.add_conditional_edges(
    "dft_agent",
    tools_condition,
)

# After tools, go back to DFT agent
workflow.add_edge("tools", "dft_agent")

# Compile the agent (no explicit checkpointer for in-memory dev use)
dft_agent = workflow.compile()
