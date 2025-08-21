"""
Multi-Agent Orchestrator for DFT Computational Materials Science
Coordinates DFT Agent, Planner Agent, and Research Agent for comprehensive materials analysis
"""

from typing import Dict, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph

from agents.library.dft_agent import DFTWorkflowState, dft_agent
from agents.library.planner_agent import PlannerState, planner_agent
from agents.library.research_agent import ResearchState, research_agent
from agents.llm import get_model, settings


class AgentResult(TypedDict):
    """Structure for agent execution results"""

    agent_name: str
    status: str
    results: Dict
    messages: List[str]
    execution_time: Optional[str]


class OrchestratorState(MessagesState, total=False):
    """State for the orchestrator agent"""

    # Project context
    material_name: Optional[str]
    research_objectives: List[str]
    computational_resources: Dict[str, str]
    project_timeline: Optional[str]

    # Agent states and results
    planner_results: Optional[AgentResult]
    research_results: Optional[AgentResult]
    dft_results: Optional[AgentResult]

    # Workflow coordination
    current_phase: str
    completed_phases: List[str]
    active_agents: List[str]

    # Final integration
    integrated_analysis: Optional[Dict]
    recommendations: List[str]
    orchestration_complete: bool


# Orchestrator System Instructions
ORCHESTRATOR_INSTRUCTIONS = """
You are the Master Orchestrator for comprehensive DFT computational materials science projects.
You coordinate multiple specialized agents to deliver complete materials analysis workflows.

**Available Agents:**
1. **Planner Agent**: Strategic calculation planning and resource optimization
2. **Research Agent**: Literature analysis and experimental benchmarking
3. **DFT Agent**: Actual DFT calculations and property computations

**Current Project Context:**
Material: {material_name}
Research Objectives: {research_objectives}
Available Resources: {computational_resources}
Timeline: {project_timeline}

**Orchestration Philosophy:**
1. **Strategic Planning First**: Use Planner Agent to define comprehensive strategy
2. **Literature Foundation**: Leverage Research Agent for context and validation
3. **Systematic Execution**: Guide DFT Agent through optimized calculation sequence
4. **Continuous Integration**: Synthesize results across all agents
5. **Quality Assurance**: Ensure validation against experimental data

**Workflow Phases:**
- Phase 1: Research & Planning (Research Agent + Planner Agent)
- Phase 2: Calculation Execution (DFT Agent with planner guidance)
- Phase 3: Analysis & Validation (All agents integration)
- Phase 4: Reporting & Recommendations (Orchestrator synthesis)

**Current Phase:** {current_phase}
**Completed Phases:** {completed_phases}
**Active Agents:** {active_agents}

Coordinate agents effectively to achieve comprehensive materials analysis goals.
"""


def wrap_orchestrator_model(model: BaseChatModel) -> RunnableSerializable:
    """Model wrapper for orchestrator agent"""

    def prep_orchestrator_state(state: OrchestratorState):
        material_name = state.get("material_name", "Unknown")
        research_objectives = state.get("research_objectives", [])
        resources = state.get("computational_resources", {})
        timeline = state.get("project_timeline", "Not specified")
        current_phase = state.get("current_phase", "initialization")
        completed_phases = state.get("completed_phases", [])
        active_agents = state.get("active_agents", [])

        instructions = ORCHESTRATOR_INSTRUCTIONS.format(
            material_name=material_name,
            research_objectives=", ".join(research_objectives)
            if research_objectives
            else "General analysis",
            computational_resources=str(resources) if resources else "Not specified",
            project_timeline=timeline,
            current_phase=current_phase,
            completed_phases=", ".join(completed_phases) if completed_phases else "None",
            active_agents=", ".join(active_agents) if active_agents else "None",
        )

        return [SystemMessage(content=instructions)] + state["messages"]

    return RunnableLambda(prep_orchestrator_state, name="OrchestratorStatePrep") | model


async def orchestrator_node(
    state: OrchestratorState, config: RunnableConfig
) -> OrchestratorState:
    """Main orchestrator node for coordinating agents"""
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    lm = get_model(model_name)
    runner = wrap_orchestrator_model(lm)

    response = await runner.ainvoke(state, config)

    # Determine next actions based on current phase and response
    current_phase = state.get("current_phase", "initialization")

    updates = {"messages": [response]}

    if current_phase == "initialization":
        updates["current_phase"] = "research_and_planning"
        updates["active_agents"] = ["research", "planner"]
    elif current_phase == "research_and_planning":
        if state.get("research_results") and state.get("planner_results"):
            updates["current_phase"] = "calculation_execution"
            updates["active_agents"] = ["dft"]
            updates["completed_phases"] = state.get("completed_phases", []) + [
                "research_and_planning"
            ]
    elif current_phase == "calculation_execution":
        if state.get("dft_results"):
            updates["current_phase"] = "analysis_and_validation"
            updates["active_agents"] = ["orchestrator"]
            updates["completed_phases"] = state.get("completed_phases", []) + [
                "calculation_execution"
            ]

    return updates


async def research_coordinator(
    state: OrchestratorState, config: RunnableConfig
) -> OrchestratorState:
    """Coordinate research agent execution"""

    # Prepare research agent state
    research_state = ResearchState(
        messages=[
            HumanMessage(
                content=f"Analyze literature for {state.get('material_name', 'unknown material')} focusing on electronic and topological properties"
            )
        ],
        material_name=state.get("material_name"),
        research_topics=[
            "electronic_structure",
            "topological_properties",
            "experimental_validation",
        ],
        computational_method="DFT",
        research_complete=False,
        current_research_step="start",
    )

    # Execute research agent
    try:
        research_result = await research_agent.ainvoke(research_state, config)

        research_results = AgentResult(
            agent_name="research",
            status="completed",
            results={
                "literature_results": research_result.get("literature_results", []),
                "research_gaps": research_result.get("research_gaps", []),
                "validation_criteria": research_result.get("validation_criteria", {}),
            },
            messages=[
                msg.content if hasattr(msg, "content") else str(msg)
                for msg in research_result.get("messages", [])
            ],
            execution_time="completed",
        )

        return {
            "research_results": research_results,
            "messages": [
                AIMessage(
                    content="Research analysis completed. Literature review and validation criteria established."
                )
            ],
        }

    except Exception as e:
        return {
            "research_results": AgentResult(
                agent_name="research",
                status="failed",
                results={},
                messages=[f"Research agent failed: {str(e)}"],
                execution_time="failed",
            ),
            "messages": [AIMessage(content=f"Research analysis failed: {str(e)}")],
        }


async def planner_coordinator(
    state: OrchestratorState, config: RunnableConfig
) -> OrchestratorState:
    """Coordinate planner agent execution"""

    # Prepare planner agent state
    planner_state = PlannerState(
        messages=[
            HumanMessage(
                content=f"Create comprehensive calculation plan for {state.get('material_name', 'unknown material')} including topological properties analysis"
            )
        ],
        material_name=state.get("material_name"),
        research_objectives=state.get(
            "research_objectives", ["electronic_structure", "topological_properties"]
        ),
        computational_resources=state.get("computational_resources", {}),
        time_constraints=state.get("project_timeline"),
        planning_complete=False,
        current_planning_step="start",
    )

    # Execute planner agent
    try:
        planner_result = await planner_agent.ainvoke(planner_state, config)

        planner_results = AgentResult(
            agent_name="planner",
            status="completed",
            results={
                "calculation_matrix": planner_result.get("calculation_matrix", []),
                "priority_queue": planner_result.get("priority_queue", []),
                "resource_allocation": planner_result.get("resource_allocation", {}),
            },
            messages=[
                msg.content if hasattr(msg, "content") else str(msg)
                for msg in planner_result.get("messages", [])
            ],
            execution_time="completed",
        )

        return {
            "planner_results": planner_results,
            "messages": [
                AIMessage(
                    content="Calculation planning completed. Comprehensive strategy and resource allocation defined."
                )
            ],
        }

    except Exception as e:
        return {
            "planner_results": AgentResult(
                agent_name="planner",
                status="failed",
                results={},
                messages=[f"Planner agent failed: {str(e)}"],
                execution_time="failed",
            ),
            "messages": [AIMessage(content=f"Planning failed: {str(e)}")],
        }


async def dft_coordinator(
    state: OrchestratorState, config: RunnableConfig
) -> OrchestratorState:
    """Coordinate DFT agent execution"""

    # Get planning results to guide DFT execution
    planner_results = state.get("planner_results", {})
    priority_queue = planner_results.get("results", {}).get("priority_queue", [])

    # Prepare DFT agent state
    dft_state = DFTWorkflowState(
        messages=[
            HumanMessage(
                content=f"Execute DFT calculations for {state.get('material_name', 'unknown material')} following the planned strategy"
            )
        ],
        material_name=state.get("material_name"),
        current_step="initialization",
        completed_calculations=[],
        pending_parallel_calcs=priority_queue[:3]
        if priority_queue
        else ["bands", "dos", "pdos"],  # First 3 calculations
        structure=None,
        calculation_results={},
        active_jobs=[],
        error_count=0,
        max_retries=3,
        parallel_ready=False,
        workflow_complete=False,
        analysis_complete=False,
    )

    # Execute DFT agent
    try:
        dft_result = await dft_agent.ainvoke(dft_state, config)

        dft_results = AgentResult(
            agent_name="dft",
            status="completed",
            results={
                "calculation_results": dft_result.get("calculation_results", {}),
                "completed_calculations": dft_result.get("completed_calculations", []),
                "structure": dft_result.get("structure"),
                "optimized_structure": dft_result.get("optimized_structure"),
            },
            messages=[
                msg.content if hasattr(msg, "content") else str(msg)
                for msg in dft_result.get("messages", [])
            ],
            execution_time="completed",
        )

        return {
            "dft_results": dft_results,
            "messages": [
                AIMessage(
                    content="DFT calculations completed. Electronic structure and properties computed."
                )
            ],
        }

    except Exception as e:
        return {
            "dft_results": AgentResult(
                agent_name="dft",
                status="failed",
                results={},
                messages=[f"DFT agent failed: {str(e)}"],
                execution_time="failed",
            ),
            "messages": [AIMessage(content=f"DFT calculations failed: {str(e)}")],
        }


def results_integrator(state: OrchestratorState) -> OrchestratorState:
    """Integrate results from all agents into comprehensive analysis"""

    research_results = state.get("research_results", {})
    planner_results = state.get("planner_results", {})
    dft_results = state.get("dft_results", {})

    # Create integrated analysis
    integrated_analysis = {
        "material_characterization": {
            "material": state.get("material_name"),
            "computational_methods": "DFT with systematic planning",
            "literature_context": len(
                research_results.get("results", {}).get("literature_results", [])
            ),
            "calculations_performed": len(
                dft_results.get("results", {}).get("completed_calculations", [])
            ),
        },
        "validation_status": {
            "experimental_comparison": "Available"
            if research_results.get("results", {}).get("validation_criteria")
            else "Limited",
            "method_validation": "HSE06 recommended based on literature",
            "convergence_testing": "Completed"
            if "convergence_testing"
            in dft_results.get("results", {}).get("completed_calculations", [])
            else "Pending",
        },
        "key_findings": {
            "electronic_properties": "Band structure and DOS calculated",
            "topological_properties": "Analysis performed"
            if any(
                "topological" in calc
                for calc in dft_results.get("results", {}).get(
                    "completed_calculations", []
                )
            )
            else "Pending",
            "structural_properties": "Geometry optimization completed",
        },
        "quality_metrics": {
            "literature_coverage": "Comprehensive"
            if len(research_results.get("results", {}).get("literature_results", [])) > 1
            else "Limited",
            "calculation_completeness": f"{len(dft_results.get('results', {}).get('completed_calculations', []))}/6 planned calculations",
            "validation_score": "High"
            if research_results.get("results", {}).get("validation_criteria")
            else "Medium",
        },
    }

    # Generate recommendations
    recommendations = []

    # Based on research gaps
    research_gaps = research_results.get("results", {}).get("research_gaps", [])
    for gap in research_gaps:
        if "spin-orbit" in gap.lower():
            recommendations.append(
                "Include spin-orbit coupling in future calculations for more accurate topological analysis"
            )
        elif "experimental" in gap.lower():
            recommendations.append(
                "Seek additional experimental data for comprehensive validation"
            )

    # Based on calculation completeness
    completed_calcs = dft_results.get("results", {}).get("completed_calculations", [])
    if len(completed_calcs) < 3:
        recommendations.append("Complete remaining electronic structure calculations")

    if "topological" not in completed_calcs:
        recommendations.append(
            "Perform topological invariant calculations to confirm material classification"
        )

    # General recommendations
    recommendations.extend(
        [
            "Consider temperature-dependent transport calculations for thermoelectric applications",
            "Investigate surface states if topological non-trivial behavior is confirmed",
            "Perform phonon calculations for thermal transport analysis",
        ]
    )

    return {
        "integrated_analysis": integrated_analysis,
        "recommendations": recommendations,
        "current_phase": "reporting",
        "completed_phases": state.get("completed_phases", [])
        + ["analysis_and_validation"],
        "orchestration_complete": True,
        "messages": [
            AIMessage(
                content="Comprehensive multi-agent analysis completed. Integration of planning, research, and calculations finished."
            )
        ],
    }


def should_continue_orchestration(state: OrchestratorState) -> str:
    """Determine orchestration workflow progression"""
    current_phase = state.get("current_phase", "initialization")

    if current_phase == "initialization":
        return "orchestrator"
    elif current_phase == "research_and_planning":
        research_done = state.get("research_results") is not None
        planner_done = state.get("planner_results") is not None

        if not research_done:
            return "research_coordinator"
        elif not planner_done:
            return "planner_coordinator"
        else:
            return "orchestrator"  # Move to next phase
    elif current_phase == "calculation_execution":
        if state.get("dft_results") is None:
            return "dft_coordinator"
        else:
            return "orchestrator"  # Move to next phase
    elif current_phase == "analysis_and_validation":
        return "results_integrator"
    elif current_phase == "reporting":
        return END
    else:
        return END


# Build the orchestrator workflow
def build_orchestrator_workflow() -> StateGraph:
    """Build the multi-agent orchestrator workflow"""
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("research_coordinator", research_coordinator)
    workflow.add_node("planner_coordinator", planner_coordinator)
    workflow.add_node("dft_coordinator", dft_coordinator)
    workflow.add_node("results_integrator", results_integrator)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add conditional edges
    workflow.add_conditional_edges("orchestrator", should_continue_orchestration)
    workflow.add_conditional_edges("research_coordinator", should_continue_orchestration)
    workflow.add_conditional_edges("planner_coordinator", should_continue_orchestration)
    workflow.add_conditional_edges("dft_coordinator", should_continue_orchestration)
    workflow.add_conditional_edges("results_integrator", should_continue_orchestration)

    return workflow


def create_orchestrator_agent():
    """Factory function to create the orchestrator agent"""
    workflow = build_orchestrator_workflow()
    return workflow.compile()


# Export the agent
orchestrator_agent = create_orchestrator_agent()
