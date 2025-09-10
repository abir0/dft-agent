"""
Plan Object and Planning Utilities

Core planning system for DFT workflows with editable, trackable plans.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PlanStatus(Enum):
    """Status of a plan or step."""

    DRAFT = "draft"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Status of individual plan steps."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Individual step in a DFT workflow plan."""

    # Core step information
    step_id: str
    tool_name: str
    description: str
    args: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

    # Dependencies and relationships
    depends_on: List[str] = field(default_factory=list)  # step_ids this depends on
    outputs: Dict[str, str] = field(default_factory=dict)  # {output_name: reference}

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "description": self.description,
            "args": self.args,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "depends_on": self.depends_on,
            "outputs": self.outputs,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Create step from dictionary."""
        step = cls(
            step_id=data["step_id"],
            tool_name=data["tool_name"],
            description=data["description"],
            args=data.get("args", {}),
            status=StepStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            depends_on=data.get("depends_on", []),
            outputs=data.get("outputs", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )

        # Parse timestamps
        if "created_at" in data:
            step.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            step.updated_at = datetime.fromisoformat(data["updated_at"])

        return step


@dataclass
class Plan:
    """Editable DFT workflow plan with execution tracking."""

    # Core plan information
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "DFT Workflow Plan"
    description: str = ""
    goal: str = ""

    # Plan structure
    steps: List[PlanStep] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Execution state
    status: PlanStatus = PlanStatus.DRAFT
    current_step_index: int = 0

    # Context and metadata
    thread_id: Optional[str] = None
    working_directory: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Execution tracking
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    total_execution_time: float = 0.0

    def add_step(
        self,
        tool_name: str,
        description: str,
        args: Dict[str, Any] = None,
        depends_on: List[str] = None,
        step_id: str = None,
    ) -> PlanStep:
        """Add a new step to the plan."""
        if step_id is None:
            step_id = f"step_{len(self.steps) + 1}"

        step = PlanStep(
            step_id=step_id,
            tool_name=tool_name,
            description=description,
            args=args or {},
            depends_on=depends_on or [],
        )

        self.steps.append(step)
        self.updated_at = datetime.now()
        return step

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def remove_step(self, step_id: str) -> bool:
        """Remove step by ID."""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                self.steps.pop(i)
                self.updated_at = datetime.now()
                return True
        return False

    def insert_step_after(
        self,
        after_step_id: str,
        tool_name: str,
        description: str,
        args: Dict[str, Any] = None,
    ) -> Optional[PlanStep]:
        """Insert a new step after the specified step."""
        for i, step in enumerate(self.steps):
            if step.step_id == after_step_id:
                new_step = PlanStep(
                    step_id=f"step_{len(self.steps) + 1}_inserted",
                    tool_name=tool_name,
                    description=description,
                    args=args or {},
                )
                self.steps.insert(i + 1, new_step)
                self.updated_at = datetime.now()
                return new_step
        return None

    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (dependencies satisfied)."""
        ready_steps = []

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in step.depends_on:
                dep_step = self.get_step(dep_id)
                if not dep_step or dep_step.status != StepStatus.COMPLETED:
                    dependencies_met = False
                    break

            if dependencies_met:
                ready_steps.append(step)

        return ready_steps

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def mark_step_completed(self, step_id: str, result: Any = None):
        """Mark a step as completed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.COMPLETED
            step.result = result
            step.updated_at = datetime.now()
            self.updated_at = datetime.now()

    def mark_step_failed(self, step_id: str, error: str):
        """Mark a step as failed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.FAILED
            step.error = error
            step.updated_at = datetime.now()
            self.updated_at = datetime.now()

    def get_progress(self) -> Dict[str, Any]:
        """Get plan execution progress."""
        total_steps = len(self.steps)
        completed_steps = len([s for s in self.steps if s.status == StepStatus.COMPLETED])
        failed_steps = len([s for s in self.steps if s.status == StepStatus.FAILED])

        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "progress_percent": (completed_steps / total_steps * 100)
            if total_steps > 0
            else 0,
            "status": self.status.value,
        }

    def resolve_step_arguments(self, step: PlanStep) -> Dict[str, Any]:
        """Resolve step arguments, replacing references with actual values."""
        resolved_args = {}

        for key, value in step.args.items():
            if isinstance(value, str) and value.startswith("$step_"):
                # Reference to another step's output
                ref_step_id = value[1:]  # Remove $
                ref_step = self.get_step(ref_step_id)
                if ref_step and ref_step.result is not None:
                    resolved_args[key] = ref_step.result
                else:
                    # Keep original value if reference can't be resolved
                    resolved_args[key] = value
            else:
                resolved_args[key] = value

        return resolved_args

    def validate_plan(self) -> List[str]:
        """Validate the plan and return any errors."""
        errors = []

        # Check for circular dependencies
        for step in self.steps:
            if self._has_circular_dependency(step.step_id, set()):
                errors.append(f"Circular dependency detected for step {step.step_id}")

        # Check that all dependencies exist
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(
                        f"Step {step.step_id} depends on non-existent step {dep_id}"
                    )

        return errors

    def _has_circular_dependency(self, step_id: str, visited: set) -> bool:
        """Check if step has circular dependencies."""
        if step_id in visited:
            return True

        visited.add(step_id)
        step = self.get_step(step_id)
        if step:
            for dep_id in step.depends_on:
                if self._has_circular_dependency(dep_id, visited.copy()):
                    return True

        return False

    def add_log_entry(self, entry_type: str, message: str, step_id: str = None):
        """Add entry to execution log."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "message": message,
            "step_id": step_id,
        }
        self.execution_log.append(log_entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "description": self.description,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "assumptions": self.assumptions,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "thread_id": self.thread_id,
            "working_directory": self.working_directory,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "execution_log": self.execution_log,
            "total_execution_time": self.total_execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Create plan from dictionary."""
        plan = cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            title=data.get("title", "DFT Workflow Plan"),
            description=data.get("description", ""),
            goal=data.get("goal", ""),
            assumptions=data.get("assumptions", []),
            success_criteria=data.get("success_criteria", []),
            status=PlanStatus(data.get("status", "draft")),
            current_step_index=data.get("current_step_index", 0),
            thread_id=data.get("thread_id"),
            working_directory=data.get("working_directory"),
            execution_log=data.get("execution_log", []),
            total_execution_time=data.get("total_execution_time", 0.0),
        )

        # Parse timestamps
        if "created_at" in data:
            plan.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            plan.updated_at = datetime.fromisoformat(data["updated_at"])

        # Parse steps
        plan.steps = [
            PlanStep.from_dict(step_data) for step_data in data.get("steps", [])
        ]

        return plan

    def to_json(self) -> str:
        """Convert plan to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Plan":
        """Create plan from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
