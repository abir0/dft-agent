"""
Extended Planning System for Unified Agent

Extends planning capabilities to work with all task types, not just DFT.
"""

import re
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from backend.agents.llm import get_model
from backend.core import OpenAIModelName

from .plan import Plan, PlanStep


def should_use_planning(user_message: str, context_tags: set) -> bool:
    """
    Determine if planning would be beneficial for the current task.
    
    Args:
        user_message: The user's message
        context_tags: Current context tags
        
    Returns:
        True if planning should be used
    """
    message_lower = user_message.lower()

    # Explicit planning triggers
    planning_keywords = [
        "create plan", "make a plan", "plan for", "planning",
        "step by step", "workflow", "procedure", "systematic"
    ]
    if any(keyword in message_lower for keyword in planning_keywords):
        return True

    # Complex task indicators
    complex_indicators = [
        "and then", "after that", "followed by", "next",
        "multiple", "series of", "sequence", "pipeline"
    ]
    if sum(1 for ind in complex_indicators if ind in message_lower) >= 2:
        return True

    # DFT/computational workflows often benefit from planning
    if "dft" in context_tags or "computational" in context_tags:
        dft_keywords = ["convergence", "optimization", "adsorption", "calculation", "simulate"]
        if any(keyword in message_lower for keyword in dft_keywords):
            return True

    # Multi-step analysis tasks
    analysis_patterns = [
        r"compare.*and.*analyze",
        r"search.*then.*calculate",
        r"generate.*test.*optimize",
        r"download.*process.*export"
    ]
    for pattern in analysis_patterns:
        if re.search(pattern, message_lower):
            return True

    return False


def create_plan_from_request(
    user_message: str,
    context_tags: set,
    available_tools: List[str]
) -> Plan:
    """
    Create a plan from user request using LLM.
    
    Args:
        user_message: The user's request
        context_tags: Current context tags
        available_tools: List of available tool names
        
    Returns:
        A Plan object with steps
    """
    model = get_model(OpenAIModelName.GPT_4O)

    # Build context-aware prompt
    context_str = ", ".join(context_tags) if context_tags else "general"
    tools_str = ", ".join(available_tools[:20])  # Show first 20 tools to avoid token explosion

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a planning assistant. Create a detailed plan for the user's request.
        
        Current context: {context}
        Available tools (sample): {tools}
        
        Respond with a numbered list of steps. Each step should:
        1. Be clear and actionable
        2. Indicate which tools might be used
        3. Build on previous steps
        
        Format:
        1. [Step description] - Tool: [tool_name if applicable]
        2. [Step description] - Tool: [tool_name if applicable]
        etc.
        """),
        ("human", "{request}")
    ])

    chain = prompt | model

    response = chain.invoke({
        "context": context_str,
        "tools": tools_str,
        "request": user_message
    })

    # Parse response into Plan
    plan = Plan(goal=user_message, steps=[])

    lines = response.content.split("\n")
    for line in lines:
        # Match numbered steps
        match = re.match(r"^\d+\.\s+(.+?)(?:\s+-\s+Tool:\s+(.+))?$", line.strip())
        if match:
            description = match.group(1)
            tool = match.group(2) if match.group(2) else None

            step = PlanStep(
                description=description,
                tool=tool,
                status="pending"
            )
            plan.steps.append(step)

    # If no steps were parsed, create a simple plan
    if not plan.steps:
        plan.steps = [
            PlanStep(
                description="Execute the requested task",
                tool=None,
                status="pending"
            )
        ]

    return plan


def update_plan_progress(plan: Plan, completed_step_index: int) -> Plan:
    """
    Update plan progress after completing a step.
    
    Args:
        plan: Current plan
        completed_step_index: Index of completed step
        
    Returns:
        Updated plan
    """
    if plan and 0 <= completed_step_index < len(plan.steps):
        plan.steps[completed_step_index].status = "completed"

        # Mark next step as in_progress if exists
        next_index = completed_step_index + 1
        if next_index < len(plan.steps):
            plan.steps[next_index].status = "in_progress"

    return plan


def get_current_step(plan: Optional[Plan]) -> Optional[PlanStep]:
    """
    Get the current step from a plan.
    
    Args:
        plan: The plan to check
        
    Returns:
        Current step or None
    """
    if not plan:
        return None

    for step in plan.steps:
        if step.status == "in_progress":
            return step

    # If no step is in progress, find first pending
    for step in plan.steps:
        if step.status == "pending":
            return step

    return None


def format_plan_for_display(plan: Plan) -> str:
    """
    Format a plan for display to the user.
    
    Args:
        plan: The plan to format
        
    Returns:
        Formatted string representation
    """
    if not plan:
        return "No active plan."

    lines = [f"**Plan: {plan.goal}**\n"]

    for i, step in enumerate(plan.steps, 1):
        status_icon = {
            "completed": "✅",
            "in_progress": "🔄",
            "pending": "⏳"
        }.get(step.status, "❓")

        tool_str = f" [Tool: {step.tool}]" if step.tool else ""
        lines.append(f"{status_icon} {i}. {step.description}{tool_str}")

    # Add progress summary
    completed = sum(1 for s in plan.steps if s.status == "completed")
    total = len(plan.steps)
    lines.append(f"\n**Progress: {completed}/{total} steps completed**")

    return "\n".join(lines)
