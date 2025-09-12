"""
Unified Agent Implementation

Combines capabilities from chatbot and DFT agent into a single comprehensive agent.
"""

from datetime import datetime
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agents.llm import get_model
from backend.core import OpenAIModelName
from backend.utils.workspace import get_workspace_path

from .planning import (
    create_plan_from_request,
    format_plan_for_display,
    get_current_step,
    should_use_planning,
    update_plan_progress,
)
from .state import ToolExecution, UnifiedAgentState
from .tool_registry import get_unified_tool_registry

# System prompt combining best of both agents
UNIFIED_SYSTEM_PROMPT = """You are a unified AI assistant with comprehensive capabilities for both general tasks and computational materials science.

You have access to:
1. General tools: web search, calculator, Python REPL, literature search
2. Structure generation: bulk, slab, supercell creation, adsorbates
3. Quantum ESPRESSO: input generation, job submission, output analysis
4. Convergence testing: k-points, cutoffs, slab thickness, vacuum
5. Materials databases: Materials Project queries, crystal analysis
6. Data management: calculation databases, results export

Key behaviors:
- Automatically use planning for complex multi-step tasks
- Maintain context awareness (general vs computational tasks)
- Track tool usage for better decision making
- Work in thread-specific directories when handling files
- Provide clear, actionable responses

When planning is active:
- Follow the plan steps systematically
- Update progress as you complete steps
- Adapt the plan if needed based on results

Your working directory is: {working_directory}
Thread ID: {thread_id}
"""


def unified_agent_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """
    Main agent node that handles all interactions.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    # Get all tools
    tools = get_unified_tool_registry()

    # Prepare model
    model = get_model(OpenAIModelName.GPT_4O)
    model_with_tools = model.bind_tools(tools)

    # Get current message
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # Check if we should use planning
    if isinstance(last_message, HumanMessage):
        user_text = last_message.content

        # Update context tags based on message content
        context_tags = state.get("context_tags", set())
        if any(kw in user_text.lower() for kw in ["dft", "quantum", "espresso", "calculation"]):
            context_tags.add("dft")
        if any(kw in user_text.lower() for kw in ["material", "crystal", "structure", "slab"]):
            context_tags.add("materials")
        if any(kw in user_text.lower() for kw in ["search", "web", "literature"]):
            context_tags.add("web")

        # Check for planning need
        if should_use_planning(user_text, context_tags):
            if not state.get("current_plan"):
                # Create a new plan
                tool_names = [t.name for t in tools]
                plan = create_plan_from_request(user_text, context_tags, tool_names)
                state["current_plan"] = plan

                # Add plan display to messages
                plan_display = format_plan_for_display(plan)
                messages.append(AIMessage(content=f"I'll help you with that. Here's my plan:\n\n{plan_display}\n\nLet me start with the first step."))

    # Handle plan execution if active
    current_plan = state.get("current_plan")
    if current_plan:
        current_step = get_current_step(current_plan)
        if current_step:
            # Add step context to prompt
            step_context = f"\n\nCurrent plan step: {current_step.description}"
            if current_step.tool:
                step_context += f" (Suggested tool: {current_step.tool})"
        else:
            step_context = "\n\nAll plan steps completed."
    else:
        step_context = ""

    # Build prompt
    system_prompt = UNIFIED_SYSTEM_PROMPT.format(
        working_directory=state.get("working_directory", get_workspace_path(state["thread_id"])),
        thread_id=state["thread_id"]
    ) + step_context

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Generate response
    chain = prompt | model_with_tools
    response = chain.invoke({"messages": messages})

    # Track tool usage if present
    if response.tool_calls:
        tool_history = state.get("tool_history", [])
        for tool_call in response.tool_calls:
            tool_history.append(ToolExecution(
                tool_name=tool_call["name"],
                timestamp=datetime.now().isoformat(),
                success=True,  # Will be updated by tool node
                context=step_context.strip() if step_context else None
            ))
        state["tool_history"] = tool_history

    # Update state
    return {
        "messages": [response],
        "context_tags": context_tags if "context_tags" in locals() else state.get("context_tags", set()),
    }


def tool_node_with_tracking(state: UnifiedAgentState) -> Dict[str, Any]:
    """
    Tool execution node with progress tracking.
    
    Args:
        state: Current state
        
    Returns:
        Updated state
    """
    # Get tools and create tool node
    tools = get_unified_tool_registry()
    tool_executor = ToolNode(tools)

    # Inject thread_id into tool calls
    thread_id = state["thread_id"]
    messages = state["messages"]

    # Process last message if it has tool calls
    if messages and hasattr(messages[-1], "tool_calls"):
        for tool_call in messages[-1].tool_calls:
            if "args" not in tool_call:
                tool_call["args"] = {}
            tool_call["args"]["_thread_id"] = thread_id

    # Execute tools - ToolNode.invoke() expects state
    result = tool_executor.invoke(state)

    # Update plan progress if applicable
    if state.get("current_plan"):
        plan = state["current_plan"]
        current_step = get_current_step(plan)
        if current_step:
            # Check if step is complete based on tool execution
            step_index = plan.steps.index(current_step)
            plan = update_plan_progress(plan, step_index)
            result["current_plan"] = plan

    return result


def should_continue(state: UnifiedAgentState) -> Literal["tools", "end"]:
    """
    Determine whether to continue with tools or end.
    
    Args:
        state: Current state
        
    Returns:
        "tools" or "end"
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if last_message and getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


def build_unified_agent():
    """
    Build the unified agent graph.
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    graph = StateGraph(UnifiedAgentState)

    # Add nodes
    graph.add_node("agent", unified_agent_node)
    graph.add_node("tools", tool_node_with_tracking)

    # Set entry point
    graph.set_entry_point("agent")

    # Add edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    graph.add_edge("tools", "agent")

    # Compile
    return graph.compile()


# Create the agent instance
unified_agent = build_unified_agent()
