"""
DFT Planner Agent - Intelligent planning and prioritization of DFT calculations
"""

from typing import Dict, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph

from agents.llm import get_model, settings


class PlanningDecision(TypedDict):
    """Structure for planning decisions"""

    calculation_type: str
    priority: int
    dependencies: List[str]
    estimated_time: str
    resource_requirements: Dict[str, str]
    reasoning: str


class PlannerState(MessagesState, total=False):
    """State for DFT planning agent"""

    # Material information
    material_name: Optional[str]
    material_properties: Optional[Dict]
    structure_info: Optional[Dict]

    # Planning context
    research_objectives: List[str]
    computational_resources: Dict[str, str]
    time_constraints: Optional[str]

    # Generated plans
    calculation_matrix: List[PlanningDecision]
    priority_queue: List[str]
    resource_allocation: Dict[str, Dict]

    # Decision tracking
    planning_complete: bool
    current_planning_step: str


# Planning System Instructions
PLANNER_INSTRUCTIONS = """
You are an expert DFT calculation planner specializing in materials science research.
Your role is to create intelligent, efficient calculation strategies for materials like ScNiSb and similar compounds.

**Core Capabilities:**
- Research objective analysis and translation to computational tasks
- Calculation dependency mapping and sequencing
- Resource optimization and parallel execution planning
- Adaptive workflow generation based on intermediate results
- Risk assessment and contingency planning

**Current Planning Context:**
Material: {material_name}
Research Objectives: {research_objectives}
Available Resources: {computational_resources}
Time Constraints: {time_constraints}

**Planning Philosophy:**
1. **Systematic Progression**: SCF → Optimization → Electronic → Advanced Properties
2. **Parallel Optimization**: Identify independent calculations for concurrent execution
3. **Adaptive Strategy**: Plan for result-driven decision points
4. **Resource Efficiency**: Balance accuracy requirements with computational cost
5. **Quality Assurance**: Build in convergence testing and validation steps

**Special Considerations for Topological Materials:**
- Include spin-orbit coupling for heavy elements
- Plan topological invariant calculations (Z2, Chern numbers)
- Consider surface state calculations if relevant
- Include transport property calculations for applications

Generate comprehensive calculation plans with clear priorities, dependencies, and resource estimates.
"""


def wrap_planner_model(model: BaseChatModel) -> RunnableSerializable:
    """Model wrapper for planner agent"""

    def prep_planner_state(state: PlannerState):
        material_name = state.get("material_name", "Unknown")
        research_objectives = state.get("research_objectives", [])
        resources = state.get("computational_resources", {})
        time_constraints = state.get("time_constraints", "None specified")

        instructions = PLANNER_INSTRUCTIONS.format(
            material_name=material_name,
            research_objectives=", ".join(research_objectives)
            if research_objectives
            else "Not specified",
            computational_resources=str(resources) if resources else "Not specified",
            time_constraints=time_constraints,
        )

        return [SystemMessage(content=instructions)] + state["messages"]

    return RunnableLambda(prep_planner_state, name="PlannerStatePrep") | model


async def planner_node(state: PlannerState, config: RunnableConfig) -> PlannerState:
    """Main planner node for generating calculation strategies"""
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    lm = get_model(model_name)
    runner = wrap_planner_model(lm)

    response = await runner.ainvoke(state, config)

    # Analyze response for planning content
    content = response.content if hasattr(response, "content") else str(response)

    # Extract planning decisions (simplified - in practice would use structured output)
    planning_decisions = extract_planning_decisions(content, state)

    updates = {
        "messages": [response],
        "calculation_matrix": planning_decisions,
        "planning_complete": True,
        "current_planning_step": "plan_generated",
    }

    return updates


def extract_planning_decisions(
    content: str, state: PlannerState
) -> List[PlanningDecision]:
    """Extract structured planning decisions from LLM response"""
    # This is a simplified version - in practice would use structured output or parsing

    material_name = state.get("material_name", "").lower()
    decisions = []

    # Standard DFT workflow
    base_decisions = [
        PlanningDecision(
            calculation_type="structure_optimization",
            priority=1,
            dependencies=[],
            estimated_time="2-4 hours",
            resource_requirements={"cores": "8-16", "memory": "32GB"},
            reasoning="Foundation for all subsequent calculations",
        ),
        PlanningDecision(
            calculation_type="convergence_testing",
            priority=1,
            dependencies=[],
            estimated_time="1-2 hours",
            resource_requirements={"cores": "4-8", "memory": "16GB"},
            reasoning="Essential for accurate results",
        ),
        PlanningDecision(
            calculation_type="electronic_structure",
            priority=2,
            dependencies=["structure_optimization", "convergence_testing"],
            estimated_time="4-8 hours",
            resource_requirements={"cores": "16-32", "memory": "64GB"},
            reasoning="Core electronic properties calculation",
        ),
    ]

    # Add material-specific calculations
    if "scnisb" in material_name or "topological" in content.lower():
        base_decisions.append(
            PlanningDecision(
                calculation_type="topological_properties",
                priority=3,
                dependencies=["electronic_structure"],
                estimated_time="8-16 hours",
                resource_requirements={"cores": "32-64", "memory": "128GB"},
                reasoning="Critical for topological material characterization",
            )
        )

    if "transport" in content.lower() or "thermoelectric" in content.lower():
        base_decisions.append(
            PlanningDecision(
                calculation_type="transport_properties",
                priority=3,
                dependencies=["electronic_structure"],
                estimated_time="4-12 hours",
                resource_requirements={"cores": "16-32", "memory": "64GB"},
                reasoning="Required for transport and thermoelectric analysis",
            )
        )

    return base_decisions


def resource_optimizer(state: PlannerState) -> PlannerState:
    """Optimize resource allocation for planned calculations"""
    calculation_matrix = state.get("calculation_matrix", [])
    available_resources = state.get("computational_resources", {})

    # Generate resource allocation plan
    resource_plan = {}

    for calc in calculation_matrix:
        calc_type = calc["calculation_type"]
        resource_plan[calc_type] = {
            "optimal_cores": extract_optimal_cores(calc["resource_requirements"]),
            "memory_requirement": calc["resource_requirements"].get("memory", "32GB"),
            "estimated_walltime": calc["estimated_time"],
            "can_run_parallel": calc["priority"] > 2,  # Advanced calcs can be parallel
        }

    return {
        "resource_allocation": resource_plan,
        "current_planning_step": "resources_optimized",
    }


def extract_optimal_cores(resource_req: Dict[str, str]) -> int:
    """Extract optimal core count from resource requirements"""
    cores_str = resource_req.get("cores", "8")
    if "-" in cores_str:
        # Take the middle value of range
        low, high = cores_str.split("-")
        return (int(low) + int(high)) // 2
    return int(cores_str)


def workflow_sequencer(state: PlannerState) -> PlannerState:
    """Generate optimized calculation sequence"""
    calculation_matrix = state.get("calculation_matrix", [])

    # Sort by priority and dependencies
    priority_queue = []

    # Add calculations in dependency order
    for priority in [1, 2, 3, 4]:
        priority_calcs = [
            calc["calculation_type"]
            for calc in calculation_matrix
            if calc["priority"] == priority
        ]
        priority_queue.extend(priority_calcs)

    return {
        "priority_queue": priority_queue,
        "current_planning_step": "sequence_generated",
    }


def should_continue_planning(state: PlannerState) -> str:
    """Determine if planning should continue"""
    if state.get("planning_complete", False):
        return END

    current_step = state.get("current_planning_step", "start")

    if current_step == "start":
        return "planner"
    elif current_step == "plan_generated":
        return "resource_optimizer"
    elif current_step == "resources_optimized":
        return "workflow_sequencer"
    else:
        return END


# Build the planner workflow
def build_planner_workflow() -> StateGraph:
    """Build the planner agent workflow"""
    workflow = StateGraph(PlannerState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("resource_optimizer", resource_optimizer)
    workflow.add_node("workflow_sequencer", workflow_sequencer)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add conditional edges
    workflow.add_conditional_edges("planner", should_continue_planning)
    workflow.add_conditional_edges("resource_optimizer", should_continue_planning)
    workflow.add_conditional_edges("workflow_sequencer", should_continue_planning)

    return workflow


def create_planner_agent():
    """Factory function to create the planner agent"""
    workflow = build_planner_workflow()
    return workflow.compile()


# Export the agent
planner_agent = create_planner_agent()
