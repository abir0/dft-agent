from datetime import datetime
from operator import add
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.dft_tools import (
    bands_calc_tool,
    bilbao_crystal_tool,
    convergence_test_tool,
    dos_calc_tool,
    generate_structure_tool,
    get_k_path,
    materials_project_lookup,
    optimize_geometry_tool,
    pdos_calc_tool,
    qe_input_generator_tool,
    qe_to_ase_tool,
    submit_job_tool,
)
from agents.llm import get_model, settings


# Enhanced State Management
class JobInfo(TypedDict):
    """Structure for job information"""

    job_id: str
    job_type: str
    status: str
    submitted_time: str
    completion_time: Optional[str]


class CalculationResult(TypedDict):
    """Structure for storing calculation results"""

    type: str
    status: str
    data: Optional[Dict]
    error: Optional[str]
    timestamp: str


class DFTWorkflowState(MessagesState, total=False):
    """Enhanced state with workflow tracking"""

    # Use Annotated for fields that might receive multiple updates
    current_step: Annotated[str, lambda x, y: y]  # Always use latest value
    completed_calculations: Annotated[List[str], add]  # Accumulate values
    pending_parallel_calcs: Annotated[List[str], lambda x, y: y]  # Use latest

    # Structure information
    structure: Optional[Dict]
    structure_source: Optional[str]  # 'mp', 'cif', 'prototype', etc.
    material_name: Optional[str]  # For tracking specific materials like ScNiSb

    # Calculation parameters
    convergence_params: Optional[Dict]
    optimized_structure: Optional[Dict]
    calculation_config: Optional[Dict]  # SOC, magnetic, etc.

    # Calculation results
    calculation_results: Dict[str, CalculationResult]

    # Job management
    active_jobs: List[JobInfo]
    completed_jobs: List[JobInfo]

    # Error tracking
    error_count: int
    max_retries: int
    retry_strategies: Dict[str, str]

    # Workflow control
    parallel_ready: bool
    workflow_complete: bool
    analysis_complete: bool


# Enhanced System Instructions
SYSTEM_INSTRUCTIONS = """
You are an expert DFT (Density Functional Theory) computational materials scientist.
You orchestrate complex quantum mechanical calculations with precision and efficiency.

**Your Capabilities:**
- Materials database queries (Materials Project)
- Crystal structure generation and analysis
- Quantum ESPRESSO input preparation
- Convergence testing and geometry optimization
- Electronic structure calculations (bands, DOS, pDOS)
- Advanced property calculations (topological, phonon, transport, magnetic, optical)
- High-performance computing job management
- Results analysis and interpretation

**Current Material:** {material_name}
**Workflow Philosophy:**
1. **Systematic Approach**: Follow established DFT best practices
2. **Quality Control**: Verify convergence before proceeding
3. **Efficiency**: Run independent calculations in parallel when possible
4. **Robustness**: Handle errors gracefully with intelligent retry logic
5. **Documentation**: Maintain clear records of all steps

**Current Workflow State:** {current_step}
**Completed Steps:** {completed_calculations}
**Pending Calculations:** {pending_parallel_calcs}
**Active Jobs:** {active_jobs}

Always explain your reasoning and provide context for your decisions.
If calculations fail, analyze the error and suggest corrections.
For materials like ScNiSb, consider topological properties and spin-orbit coupling effects.
"""

# Enhanced tools list with advanced property calculations
tools = [
    materials_project_lookup,
    generate_structure_tool,
    convergence_test_tool,
    optimize_geometry_tool,
    bands_calc_tool,
    dos_calc_tool,
    pdos_calc_tool,
    # topological_invariant_calc_tool,  # Available when needed
    # phonon_calc_tool,  # Available when needed
    # boltztrap_calc_tool,  # Available when needed
    # magnetic_properties_calc_tool,  # Available when needed
    # optical_properties_calc_tool,  # Available when needed
    qe_to_ase_tool,
    submit_job_tool,
    get_k_path,
    bilbao_crystal_tool,
    qe_input_generator_tool,
]


def wrap_model(model: BaseChatModel) -> RunnableSerializable:
    """Enhanced model wrapper with dynamic instructions"""
    model = model.bind_tools(tools)

    def prep_state(state: DFTWorkflowState):
        current_step = state.get("current_step", "initialization")
        completed = state.get("completed_calculations", [])
        pending = state.get("pending_parallel_calcs", [])
        material_name = state.get("material_name", "Unknown")
        active_jobs = state.get("active_jobs", [])

        instructions = SYSTEM_INSTRUCTIONS.format(
            current_step=current_step,
            completed_calculations=", ".join(completed) if completed else "None",
            pending_parallel_calcs=", ".join(pending) if pending else "None",
            material_name=material_name,
            active_jobs=len(active_jobs),
        )

        return [SystemMessage(content=instructions)] + state["messages"]

    return RunnableLambda(prep_state, name="StatePrep") | model


async def call_model(state: DFTWorkflowState, config: RunnableConfig) -> DFTWorkflowState:
    """Enhanced model calling with context awareness"""
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    lm = get_model(model_name)
    runner = wrap_model(lm)

    try:
        resp = await runner.ainvoke(state, config)

        # Update workflow tracking
        updates = {"messages": [resp]}

        # Analyze response for workflow progression
        if hasattr(resp, "tool_calls") and resp.tool_calls:
            tool_names = [tc["name"] for tc in resp.tool_calls]
            updates["current_step"] = f"executing_{tool_names[0]}"

        return updates

    except Exception as e:
        error_msg = f"Model call failed: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_step": "error_recovery",
        }


def handle_tool_error(state: DFTWorkflowState) -> DFTWorkflowState:
    """Enhanced error handling with retry logic"""
    error = state.get("error")
    error_count = state.get("error_count", 0) + 1
    max_retries = state.get("max_retries", 3)

    if error_count >= max_retries:
        error_msg = f"Max retries ({max_retries}) exceeded. Last error: {error}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_step": "failed",
            "error_count": error_count,
        }

    # Determine retry strategy based on error type
    retry_strategy = get_retry_strategy(error)

    calls = state["messages"][-1].tool_calls if state["messages"] else []
    tool_messages = [
        ToolMessage(
            content=f"Error: {error}. Retry strategy: {retry_strategy}",
            tool_call_id=tc["id"],
        )
        for tc in calls
    ]

    return {
        "messages": tool_messages,
        "error_count": error_count,
        "current_step": "retrying",
    }


def should_continue(state: DFTWorkflowState) -> str:
    """Determine if workflow should continue or end"""
    if state.get("workflow_complete", False):
        return "chatbot"

    # Check for terminal conditions
    current_step = state.get("current_step", "")
    if current_step in ["failed", "error"]:
        return "chatbot"

    # If we have completed all main calculations, switch to chatbot mode
    completed = set(state.get("completed_calculations", []))
    required_steps = {
        "structure_acquired",
        "convergence_testing",
        "geometry_optimization",
    }

    if required_steps.issubset(completed):
        calc_results = state.get("calculation_results", {})
        if len(calc_results) >= 3:  # bands, dos, pdos completed
            return "chatbot"

    return "continue_workflow"


def should_terminate(state: DFTWorkflowState) -> bool:
    """Check if the conversation should terminate"""
    messages = state.get("messages", [])

    if not messages:
        return True

    last_message = messages[-1]

    # If the last message is from AI and has no tool calls, end the conversation
    if (
        hasattr(last_message, "__class__")
        and last_message.__class__.__name__ == "AIMessage"
        and not (hasattr(last_message, "tool_calls") and last_message.tool_calls)
    ):
        return True

    return False


# Add chatbot node
async def chatbot_node(
    state: DFTWorkflowState, config: RunnableConfig
) -> DFTWorkflowState:
    """Simple chatbot for Q&A and workflow initiation"""
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    lm = get_model(model_name)

    # Bind tools to the model for the chatbot
    model_with_tools = lm.bind_tools(tools)

    # Prepare instructions for the chatbot
    current_step = state.get("current_step", "initialization")
    completed = state.get("completed_calculations", [])
    pending = state.get("pending_parallel_calcs", [])

    instructions = SYSTEM_INSTRUCTIONS.format(
        current_step=current_step,
        completed_calculations=", ".join(completed) if completed else "None",
        pending_parallel_calcs=", ".join(pending) if pending else "None",
        material_name=state.get("material_name", "Unknown"),
        active_jobs=len(state.get("active_jobs", [])),
    )

    # Prepare the messages with system instructions
    messages = [SystemMessage(content=instructions)] + state["messages"]

    response = await model_with_tools.ainvoke(messages, config)

    # Update workflow tracking based on response
    updates = {"messages": [response]}

    # Analyze response for workflow progression
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        updates["current_step"] = f"executing_{tool_names[0]}"

    return updates


def get_retry_strategy(error: str) -> str:
    """Determine appropriate retry strategy based on error type"""
    error_lower = str(error).lower()

    if "convergence" in error_lower:
        return "adjust_convergence_parameters"
    elif "memory" in error_lower or "resource" in error_lower:
        return "reduce_calculation_size"
    elif "file" in error_lower or "io" in error_lower:
        return "check_file_permissions"
    elif "network" in error_lower or "connection" in error_lower:
        return "retry_network_operation"
    else:
        return "generic_retry"


def make_enhanced_tool_node(tools_list: List):
    """Create tool node with enhanced error handling"""
    return ToolNode(tools_list).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Workflow Step Functions
def convergence_testing(state: DFTWorkflowState) -> DFTWorkflowState:
    """Handle convergence testing step"""
    if not state.get("structure"):
        return {
            "messages": [
                AIMessage(content="Error: No structure available for convergence testing")
            ],
            "current_step": "error",
        }

    return {
        "current_step": "convergence_testing",
        "messages": [
            AIMessage(
                content="Starting convergence tests for k-points and energy cutoff..."
            )
        ],
    }


def geometry_optimization(state: DFTWorkflowState) -> DFTWorkflowState:
    """Handle geometry optimization step"""
    if not state.get("convergence_params"):
        return {
            "messages": [
                AIMessage(
                    content="Warning: No convergence parameters found, using defaults for geometry optimization"
                )
            ],
            "current_step": "geometry_optimization",
        }

    return {
        "current_step": "geometry_optimization",
        "messages": [
            AIMessage(
                content="Starting geometry optimization with converged parameters..."
            )
        ],
    }


def parallel_coordinator(state: DFTWorkflowState) -> DFTWorkflowState:
    """Coordinate parallel calculations with advanced properties"""
    if not state.get("optimized_structure") and not state.get("structure"):
        return {
            "messages": [
                AIMessage(
                    content="Error: No structure available for electronic structure calculations"
                )
            ],
            "current_step": "error",
        }

    # Enhanced parallel calculations including advanced properties
    base_calcs = ["bands", "dos", "pdos"]

    # Determine which calculations to run based on material and configuration
    material_name = state.get("material_name", "").lower()
    calc_config = state.get("calculation_config", {})

    pending_calcs = base_calcs.copy()

    # Add advanced calculations based on material type and user configuration
    if "scnisb" in material_name or calc_config.get("topological", False):
        pending_calcs.append("topological")

    if calc_config.get("include_phonon", True):
        pending_calcs.append("phonon")

    if calc_config.get("include_transport", True):
        pending_calcs.append("transport")

    if calc_config.get("magnetic", False):
        pending_calcs.append("magnetic")

    if calc_config.get("optical", False):
        pending_calcs.append("optical")

    return {
        "current_step": "parallel_calculations",
        "pending_parallel_calcs": pending_calcs,
        "parallel_ready": True,
        "messages": [
            AIMessage(
                content=f"Setting up parallel calculations: {', '.join(pending_calcs)}..."
            )
        ],
    }


def workflow_router(state: DFTWorkflowState) -> str:
    """Enhanced routing logic based on workflow state"""
    if state.get("workflow_complete", False):
        return "chatbot"

    current_step = state.get("current_step", "initialization")
    completed = set(state.get("completed_calculations", []))
    pending_parallel = state.get("pending_parallel_calcs", [])

    # Error handling
    if current_step == "failed":
        return "chatbot"

    # If we have structure and workflow should proceed automatically
    if state.get("structure"):
        # Sequential workflow logic
        if "structure_acquired" not in completed:
            return "chatbot"  # Let chatbot handle next steps
        elif "convergence_testing" not in completed:
            return "convergence_testing"
        elif "geometry_optimization" not in completed:
            return "geometry_optimization"
        elif pending_parallel:
            # Route to next pending parallel calculation
            next_calc = pending_parallel[0]
            return next_calc
        elif len(state.get("calculation_results", {})) >= 3:
            return "gather_results"
        else:
            return "parallel_coordinator"

    # No structure or default case - go to chatbot
    return "chatbot"


def update_completed_calculations(
    state: DFTWorkflowState, calc_type: str
) -> DFTWorkflowState:
    """Update state when a calculation completes"""
    completed = set(state.get("completed_calculations", []))
    completed.add(calc_type)

    pending = state.get("pending_parallel_calcs", [])
    if calc_type in pending:
        pending.remove(calc_type)

    # Store calculation result
    results = state.get("calculation_results", {})
    results[calc_type] = CalculationResult(
        type=calc_type,
        status="completed",
        data={},  # Would be populated by actual tool
        error=None,
        timestamp=datetime.now().isoformat(),
    )

    return {
        "completed_calculations": list(completed),
        "pending_parallel_calcs": pending,
        "calculation_results": results,
    }


def gather_parallel_results(state: DFTWorkflowState) -> DFTWorkflowState:
    """Gather and analyze results from parallel calculations"""
    results = state.get("calculation_results", {})

    # Check completion status
    required_calcs = {"bands", "dos", "pdos"}
    completed_calcs = set(results.keys())

    if not required_calcs.issubset(completed_calcs):
        missing = required_calcs - completed_calcs
        return {
            "messages": [
                AIMessage(
                    content=f"Warning: Still waiting for calculations: {list(missing)}"
                )
            ],
            "current_step": "waiting_for_parallel",
        }

    # Generate comprehensive summary
    summary = generate_results_summary(results)

    return {
        "messages": [AIMessage(content=summary)],
        "current_step": "results_analysis",
        "parallel_ready": False,
        "workflow_complete": True,
    }


def generate_results_summary(results: Dict[str, CalculationResult]) -> str:
    """Generate a comprehensive summary of calculation results"""
    summary = "=== DFT Calculation Results Summary ===\n\n"

    for calc_type, result in results.items():
        summary += f"**{calc_type.upper()} Calculation:**\n"
        summary += f"  Status: {result['status']}\n"
        summary += f"  Timestamp: {result['timestamp']}\n"

        if result["error"]:
            summary += f"  Error: {result['error']}\n"
        elif result["data"]:
            summary += f"  Data keys: {list(result['data'].keys())}\n"

        summary += "\n"

    # Add analysis recommendations
    summary += "**Analysis & Recommendations:**\n"
    if all(r["status"] == "completed" for r in results.values()):
        summary += "✓ All calculations completed successfully\n"
        summary += "→ Ready for post-processing and visualization\n"
        summary += (
            "→ Consider running additional analysis: optical properties, phonons, etc.\n"
        )
    else:
        failed = [k for k, v in results.items() if v["status"] == "failed"]
        summary += f"⚠ Failed calculations: {failed}\n"
        summary += "→ Consider parameter adjustment and retry\n"

    return summary


# Enhanced calculation nodes that update state
def bands_calculation_node(state: DFTWorkflowState) -> DFTWorkflowState:
    """Bands calculation with state update"""
    # This would normally call the actual tool
    # For now, simulate completion
    updates = update_completed_calculations(state, "bands")
    updates["messages"] = [AIMessage(content="Band structure calculation completed")]
    return updates


def dos_calculation_node(state: DFTWorkflowState) -> DFTWorkflowState:
    """DOS calculation with state update"""
    updates = update_completed_calculations(state, "dos")
    updates["messages"] = [AIMessage(content="Density of states calculation completed")]
    return updates


def pdos_calculation_node(state: DFTWorkflowState) -> DFTWorkflowState:
    """pDOS calculation with state update"""
    updates = update_completed_calculations(state, "pdos")
    updates["messages"] = [
        AIMessage(content="Projected density of states calculation completed")
    ]
    return updates


def route_chatbot(state: DFTWorkflowState) -> str:
    """Route from chatbot based on tool calls and workflow state"""
    last_message = state["messages"][-1]

    # If the last message has tool calls, route to the appropriate tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]

        tool_to_node = {
            "materials_project_lookup": "mp_lookup",
            "generate_structure_tool": "structure_gen",
            "bilbao_crystal_tool": "structure_gen",
            "get_k_path": "structure_gen",
            "convergence_test_tool": "scf_converge",
            "optimize_geometry_tool": "geom_opt",
            "bands_calc_tool": "bands_tool",
            "dos_calc_tool": "dos_tool",
            "pdos_calc_tool": "pdos_tool",
            "qe_to_ase_tool": "qe2ase",
            "submit_job_tool": "submit_job",
            "qe_input_generator_tool": "qe_input_gen",
        }

        return tool_to_node.get(tool_name, END)

    # If AI just completed its response without tool calls, end the conversation
    if should_terminate(state):
        return END

    # Otherwise continue the conversation
    return "chatbot"


def route_tools(state: DFTWorkflowState) -> str:
    """Route tool calls to appropriate nodes"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]

        # Map tool names to node names
        tool_to_node = {
            "materials_project_lookup": "mp_lookup",
            "generate_structure_tool": "structure_gen",
            "bilbao_crystal_tool": "structure_gen",
            "get_k_path": "structure_gen",
            "convergence_test_tool": "scf_converge",
            "optimize_geometry_tool": "geom_opt",
            "bands_calc_tool": "bands_tool",
            "dos_calc_tool": "dos_tool",
            "pdos_calc_tool": "pdos_tool",
            "qe_to_ase_tool": "qe2ase",
            "submit_job_tool": "submit_job",
            "qe_input_generator_tool": "qe_input_gen",
        }

        return tool_to_node.get(tool_name, "chatbot")

    # No tool calls - continue with workflow routing
    return workflow_router(state)


# Build the workflow
def build_workflow() -> StateGraph:
    """Build the enhanced workflow graph without Send API"""
    workflow = StateGraph(DFTWorkflowState)

    # Core nodes
    workflow.add_node("planner", call_model)
    workflow.add_node("convergence_testing", convergence_testing)
    workflow.add_node("geometry_optimization", geometry_optimization)
    workflow.add_node("parallel_coordinator", parallel_coordinator)
    workflow.add_node("gather_results", gather_parallel_results)

    # Add chatbot node
    workflow.add_node("chatbot", chatbot_node)

    # Tool nodes
    workflow.add_node("mp_lookup", make_enhanced_tool_node([materials_project_lookup]))
    workflow.add_node(
        "structure_gen",
        make_enhanced_tool_node(
            [generate_structure_tool, bilbao_crystal_tool, get_k_path]
        ),
    )
    workflow.add_node("scf_converge", make_enhanced_tool_node([convergence_test_tool]))
    workflow.add_node("geom_opt", make_enhanced_tool_node([optimize_geometry_tool]))

    # Calculation nodes (enhanced with state management)
    workflow.add_node("bands", bands_calculation_node)
    workflow.add_node("dos", dos_calculation_node)
    workflow.add_node("pdos", pdos_calculation_node)

    workflow.add_node("bands_tool", make_enhanced_tool_node([bands_calc_tool]))
    workflow.add_node("dos_tool", make_enhanced_tool_node([dos_calc_tool]))
    workflow.add_node("pdos_tool", make_enhanced_tool_node([pdos_calc_tool]))
    workflow.add_node("qe_input_gen", make_enhanced_tool_node([qe_input_generator_tool]))

    # Utility nodes
    workflow.add_node("qe2ase", make_enhanced_tool_node([qe_to_ase_tool]))
    workflow.add_node("submit_job", make_enhanced_tool_node([submit_job_tool]))

    # Entry point - start with chatbot for user interaction
    workflow.set_entry_point("chatbot")

    # Chatbot routing
    workflow.add_conditional_edges("chatbot", route_chatbot)

    # Main workflow routing from planner
    workflow.add_conditional_edges("planner", route_tools)

    workflow.add_conditional_edges(
        "gather_results",
        should_continue,
        {"chatbot": "chatbot", "continue_workflow": "chatbot"},
    )

    # Workflow step edges - go back to chatbot for user interaction
    workflow.add_edge("convergence_testing", "chatbot")
    workflow.add_edge("geometry_optimization", "chatbot")
    workflow.add_edge("parallel_coordinator", "chatbot")

    # Tool edges back to chatbot for user feedback
    workflow.add_edge("mp_lookup", "chatbot")
    workflow.add_edge("structure_gen", "chatbot")
    workflow.add_edge("scf_converge", "chatbot")
    workflow.add_edge("geom_opt", "chatbot")

    # Calculation edges
    workflow.add_conditional_edges("bands", workflow_router)
    workflow.add_conditional_edges("dos", workflow_router)
    workflow.add_conditional_edges("pdos", workflow_router)

    workflow.add_edge("bands_tool", "chatbot")
    workflow.add_edge("dos_tool", "chatbot")
    workflow.add_edge("pdos_tool", "chatbot")
    workflow.add_edge("qe_input_gen", "chatbot")

    # Results gathering
    workflow.add_edge("gather_results", "chatbot")

    # Utility edges
    workflow.add_edge("qe2ase", "chatbot")
    workflow.add_edge("submit_job", "chatbot")

    return workflow


# Create the enhanced DFT agent
def create_dft_agent():
    """Factory function to create the enhanced DFT agent"""
    workflow = build_workflow()
    return workflow.compile()


# Export the agent
dft_agent = create_dft_agent()
