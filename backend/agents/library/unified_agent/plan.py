"""
Plan data structures for the unified agent.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlanStep:
    """A single step in a plan."""
    description: str
    tool: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed


@dataclass
class Plan:
    """A multi-step plan for completing a task."""
    goal: str
    steps: List[PlanStep]
