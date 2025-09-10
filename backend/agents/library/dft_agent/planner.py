"""
Planner Agent Implementation

A comprehensive planner agent that generates, executes, and modifies DFT workflow plans.
This agent serves as the main entry point for users, routing between planning,
execution, and chat capabilities.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph

from backend.agents.llm import get_model, settings

from .agent import custom_tool_node, dft_tools, initialize_dft_state
from .plan import Plan, PlanStatus, StepStatus
from .tool_registry import TOOL_REGISTRY


class PlannerState(MessagesState, total=False):
    """State for the planner agent that tracks plans and execution."""

    # Core DFT state (inherited from original agent)
    working_directory: str
    thread_id: str
    current_structures: Dict[str, str]
    last_calculation: Optional[Dict[str, Any]]
    current_workflow: Optional[str]
    workflow_step: int

    # Planning state
    current_plan: Optional[Plan]
    plan_history: List[Plan]
    planner_mode: str  # "planning", "executing", "chat", "review"

    # Execution tracking
    execution_errors: List[str]
    auto_execute: bool


def initialize_planner_state(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize planner agent state."""
    base_state = initialize_dft_state(config)

    planner_state = {
        **base_state,
        "current_plan": None,
        "plan_history": [],
        "planner_mode": "planning",
        "execution_errors": [],
        "auto_execute": False,
    }

    return planner_state


# System prompts for different modes
PLANNER_SYSTEM_PROMPT = f"""
You are an expert DFT (Density Functional Theory) workflow planner and execution agent.
Today's date is {datetime.now().strftime("%B %d, %Y")}.

You have multiple capabilities:

1. **PLANNING MODE**: Generate comprehensive, executable DFT workflow plans
2. **EXECUTION MODE**: Execute plans step-by-step using available tools
3. **CHAT MODE**: Answer questions and provide DFT guidance
4. **REVIEW MODE**: Analyze results and modify plans as needed

AVAILABLE TOOLS AND CAPABILITIES:
- Structure generation and manipulation (bulk, surfaces, supercells)
- Quantum ESPRESSO interface (input generation, execution, parsing)
- VASP interface (input generation, execution, parsing) 
- Convergence testing (k-points, cutoffs, slab thickness)
- Database management and result storage
- Job execution and monitoring

PLANNING GUIDELINES:
- Break complex workflows into logical, executable steps
- Use proper tool names from the registry
- Handle dependencies between steps correctly
- Include validation and error checking steps
- Consider computational efficiency and best practices

EXECUTION GUIDELINES:
- Execute steps in dependency order
- Handle errors gracefully and suggest modifications
- Track progress and provide status updates
- Store intermediate results for reference

MODES:
- **planning**: User wants to create or modify a plan
- **executing**: Currently executing a plan step-by-step
- **chat**: General conversation, questions, or guidance
- **review**: Analyzing results and planning next steps

Always be helpful, accurate, and focused on computational materials science workflows.
"""


def wrap_planner_model(
    model: BaseChatModel,
) -> RunnableSerializable[PlannerState, AIMessage]:
    """Wrap model with planner-specific preprocessing."""
    model = model.bind_tools(dft_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + state["messages"],
        name="PlannerStateModifier",
    )
    return preprocessor | model


async def planner_chat_node(state: PlannerState, config: RunnableConfig) -> PlannerState:
    """Main planner node that handles planning, chat, and coordination."""

    # Initialize state if needed
    if "working_directory" not in state or "thread_id" not in state:
        state.update(initialize_planner_state(config))

    # Determine current mode based on context
    last_message = state["messages"][-1] if state["messages"] else None
    current_mode = determine_planner_mode(state, last_message)
    state["planner_mode"] = current_mode

    # Get model and create runnable
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    m = get_model(model_name)
    planner_runnable = wrap_planner_model(m)

    # Add context about current plan and mode to the conversation
    context_message = create_context_message(state)

    # Create messages for the model
    model_messages = state["messages"] + ([context_message] if context_message else [])
    temp_state = {**state, "messages": model_messages}

    # Get model response
    response = await planner_runnable.ainvoke(temp_state, config)

    return {"messages": [response]}


def determine_planner_mode(state: PlannerState, last_message: BaseMessage) -> str:
    """Determine what mode the planner should be in based on context."""
    if not last_message:
        return "planning"

    message_content = (
        last_message.content.lower() if hasattr(last_message, "content") else ""
    )

    # Check for explicit mode requests
    if any(
        word in message_content for word in ["plan", "workflow", "steps", "create plan"]
    ):
        return "planning"
    elif any(word in message_content for word in ["execute", "run", "start", "continue"]):
        return "executing"
    elif any(
        word in message_content for word in ["results", "analyze", "review", "modify"]
    ):
        return "review"

    # Check current plan status
    current_plan = state.get("current_plan")
    if current_plan:
        if current_plan.status == PlanStatus.READY:
            return "executing"
        elif current_plan.status in [PlanStatus.EXECUTING, PlanStatus.PAUSED]:
            return "executing"
        elif current_plan.status == PlanStatus.COMPLETED:
            return "review"

    # Default to chat mode for general questions
    return "chat"


def create_context_message(state: PlannerState) -> Optional[SystemMessage]:
    """Create context message about current plan and state."""
    current_plan = state.get("current_plan")
    mode = state.get("planner_mode", "planning")

    context_parts = []

    # Add mode information
    context_parts.append(f"CURRENT MODE: {mode.upper()}")

    # Add plan information
    if current_plan:
        progress = current_plan.get_progress()
        context_parts.append(f"CURRENT PLAN: {current_plan.title}")
        context_parts.append(f"GOAL: {current_plan.goal}")
        context_parts.append(
            f"PROGRESS: {progress['completed_steps']}/{progress['total_steps']} steps completed ({progress['progress_percent']:.1f}%)"
        )
        context_parts.append(f"STATUS: {current_plan.status.value}")

        # Add ready steps info
        ready_steps = current_plan.get_ready_steps()
        if ready_steps:
            context_parts.append(f"READY TO EXECUTE: {len(ready_steps)} steps")
    else:
        context_parts.append("CURRENT PLAN: None")

    # Add workspace info
    working_dir = state.get("working_directory")
    if working_dir:
        context_parts.append(f"WORKSPACE: {working_dir}")

    # Add structure info
    structures = state.get("current_structures", {})
    if structures:
        context_parts.append(f"ACTIVE STRUCTURES: {list(structures.keys())}")

    context = "\\n".join(context_parts)
    return SystemMessage(content=f"CONTEXT:\\n{context}") if context_parts else None


async def plan_executor_node(state: PlannerState, config: RunnableConfig) -> PlannerState:
    """Execute the next ready step in the current plan."""

    current_plan = state.get("current_plan")
    if not current_plan:
        return {
            "messages": [
                AIMessage(
                    content="No active plan to execute. Please create a plan first."
                )
            ]
        }

    # Get ready steps
    ready_steps = current_plan.get_ready_steps()
    if not ready_steps:
        completed_steps = len(
            [s for s in current_plan.steps if s.status == StepStatus.COMPLETED]
        )
        total_steps = len(current_plan.steps)

        if completed_steps == total_steps:
            current_plan.status = PlanStatus.COMPLETED
            message = f"✅ Plan execution completed! All {total_steps} steps finished successfully."
        else:
            message = "⏸️ No steps ready to execute. Check plan dependencies or previous step failures."

        return {"messages": [AIMessage(content=message)]}

    # Execute the first ready step
    step = ready_steps[0]
    step.status = StepStatus.EXECUTING
    current_plan.add_log_entry(
        "execution", f"Starting step: {step.description}", step.step_id
    )

    try:
        # Get tool function
        tool_func = TOOL_REGISTRY.get(step.tool_name)
        if not tool_func:
            raise ValueError(f"Tool '{step.tool_name}' not found in registry")

        # Resolve step arguments
        resolved_args = current_plan.resolve_step_arguments(step)

        # Add thread_id if needed
        if state.get("thread_id"):
            resolved_args["_thread_id"] = state["thread_id"]

        # Execute the tool
        import time

        start_time = time.time()
        result = tool_func.func(**resolved_args)
        execution_time = time.time() - start_time

        # Mark step as completed
        step.status = StepStatus.COMPLETED
        step.result = result
        step.execution_time = execution_time
        current_plan.total_execution_time += execution_time

        current_plan.add_log_entry(
            "success", f"Completed step: {step.description}", step.step_id
        )

        message = f"✅ Step completed: {step.description}\\n\\nResult: {result}"

        # Check if there are more ready steps
        remaining_ready = current_plan.get_ready_steps()
        if remaining_ready:
            message += f"\\n\\n📋 {len(remaining_ready)} more steps ready to execute."

        return {"messages": [AIMessage(content=message)]}

    except Exception as e:
        # Mark step as failed
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.retry_count += 1

        current_plan.add_log_entry(
            "error", f"Step failed: {step.description} - {str(e)}", step.step_id
        )

        error_message = f"❌ Step failed: {step.description}\\n\\nError: {str(e)}"

        # Check if step can be retried
        if step.retry_count < step.max_retries:
            step.status = StepStatus.PENDING  # Reset for retry
            error_message += (
                f"\\n\\n🔄 Will retry ({step.retry_count}/{step.max_retries})"
            )
        else:
            error_message += "\\n\\n🚫 Max retries reached. Manual intervention required."

        return {"messages": [AIMessage(content=error_message)]}


def route_planner(state: PlannerState) -> str:
    """Route between different planner nodes."""
    messages = state["messages"]
    if not messages:
        return "planner_chat"

    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check current mode
    mode = state.get("planner_mode", "planning")

    # Check for explicit execution requests
    if hasattr(last_message, "content"):
        content = last_message.content.lower()

        if any(
            word in content for word in ["execute next", "run next", "continue execution"]
        ):
            return "plan_executor"

        if "execute plan" in content or "start execution" in content:
            current_plan = state.get("current_plan")
            if current_plan and current_plan.status == PlanStatus.READY:
                return "plan_executor"

    # Check if there's an active plan being executed automatically
    if state.get("auto_execute", False):
        current_plan = state.get("current_plan")
        if current_plan and current_plan.status == PlanStatus.EXECUTING:
            ready_steps = current_plan.get_ready_steps()
            if ready_steps:
                return "plan_executor"

    return "planner_chat"


def create_plan_from_llm_response(
    response_content: str, state: PlannerState
) -> Optional[Plan]:
    """Extract and create a plan from LLM response."""
    try:
        # Try to find JSON in the response
        import re

        json_match = re.search(r"\\{.*\\}", response_content, re.DOTALL)
        if json_match:
            plan_data = json.loads(json_match.group())

            # Create plan object
            plan = Plan(
                title=plan_data.get("title", "DFT Workflow Plan"),
                description=plan_data.get("description", ""),
                goal=plan_data.get("goal", ""),
                assumptions=plan_data.get("assumptions", []),
                success_criteria=plan_data.get("success_criteria", []),
                thread_id=state.get("thread_id"),
                working_directory=state.get("working_directory"),
            )

            # Add steps
            for i, step_data in enumerate(plan_data.get("steps", [])):
                plan.add_step(
                    tool_name=step_data["tool"],
                    description=step_data.get(
                        "explain", step_data.get("description", f"Step {i + 1}")
                    ),
                    args=step_data.get("args", {}),
                    depends_on=step_data.get("depends_on", []),
                )

            plan.status = PlanStatus.READY
            return plan

    except Exception as e:
        print(f"Failed to parse plan from LLM response: {e}")

    return None


# Create the planner workflow graph
def create_planner_agent():
    """Create the planner agent workflow."""

    workflow = StateGraph(PlannerState)

    # Add nodes
    workflow.add_node("planner_chat", planner_chat_node)
    workflow.add_node("plan_executor", plan_executor_node)
    workflow.add_node("tools", custom_tool_node)

    # Set entry point
    workflow.set_entry_point("planner_chat")

    # Add routing
    workflow.add_conditional_edges(
        "planner_chat",
        route_planner,
        {
            "planner_chat": "__end__",
            "plan_executor": "plan_executor",
            "tools": "tools",
        },
    )

    # After plan execution, go back to chat
    workflow.add_edge("plan_executor", "planner_chat")

    # After tools, go back to chat
    workflow.add_edge("tools", "planner_chat")

    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())


# Create the main planner agent instance
planner_agent = create_planner_agent()
