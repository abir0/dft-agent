"""
Unified Agent State Schema

Streamlined state that supports both general tasks and computational workflows.
"""

from typing import Annotated, List, Optional, Set, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .plan import Plan


class ToolExecution(TypedDict):
    """Record of a tool execution."""
    tool_name: str
    timestamp: str
    success: bool
    context: Optional[str]


class UnifiedAgentState(TypedDict):
    """Streamlined state for the unified agent."""

    # Core conversation state
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str

    # Working environment
    working_directory: str

    # Planning system (works for all task types)
    current_plan: Optional[Plan]

    # Tool execution tracking
    tool_history: List[ToolExecution]

    # Context tags for smart behavior
    context_tags: Set[str]  # e.g., {"dft", "materials", "general", "web", "analysis"}

    # Optional: Track current workflow type
    workflow_type: Optional[str]  # e.g., "general", "dft_calculation", "materials_search"
