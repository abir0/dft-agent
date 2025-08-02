"""
Research and Literature Review Agent for DFT materials science
Provides comprehensive literature analysis and experimental data context
"""

from typing import Dict, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph

from agents.llm import get_model, settings


class LiteratureResult(TypedDict):
    """Structure for literature search results"""

    title: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str]
    key_findings: List[str]
    relevance_score: float
    experimental_data: Optional[Dict]


class ResearchState(MessagesState, total=False):
    """State for research agent"""

    # Research context
    material_name: Optional[str]
    research_topics: List[str]
    computational_method: Optional[str]

    # Literature results
    literature_results: List[LiteratureResult]
    experimental_benchmarks: Dict[str, Dict]
    theoretical_predictions: Dict[str, Dict]

    # Analysis results
    research_gaps: List[str]
    methodological_recommendations: List[str]
    validation_criteria: Dict[str, str]

    # Completion tracking
    research_complete: bool
    current_research_step: str


# Research System Instructions
RESEARCH_INSTRUCTIONS = """
You are an expert materials science research analyst specializing in computational DFT studies.
Your role is to provide comprehensive literature analysis, experimental data context, and methodological guidance.

**Core Capabilities:**
- Literature search and synthesis for materials science publications
- Experimental data compilation and benchmarking
- Computational method validation and recommendations
- Research gap identification and future direction suggestions
- Cross-validation between theory and experiment

**Current Research Context:**
Material: {material_name}
Research Topics: {research_topics}
Computational Method: {computational_method}

**Research Philosophy:**
1. **Comprehensive Coverage**: Include both theoretical and experimental studies
2. **Critical Analysis**: Evaluate methodology quality and reproducibility
3. **Benchmarking**: Identify reliable experimental references
4. **Method Validation**: Compare computational approaches and accuracy
5. **Innovation Identification**: Highlight novel techniques and findings

**Special Focus Areas:**
- Topological materials and their electronic properties
- Thermoelectric materials and transport properties
- DFT accuracy for specific material classes
- Experimental validation methods
- Emerging computational techniques

Provide thorough, critical analysis with clear recommendations for computational studies.
"""


@tool("literature_search_tool")
def literature_search_tool(material_name: str, keywords: List[str]) -> str:
    """
    Search literature for materials science publications.
    Args:
        material_name: Name of the material to search for
        keywords: List of relevant keywords for the search
    Returns:
        JSON string with literature results
    """
    # Simulated literature search - in practice would use APIs like CrossRef, arXiv, etc.

    search_results = []

    return search_results


@tool("experimental_data_search_tool")
def experimental_data_search_tool(material_name: str, property_type: str) -> str:
    """
    Search for experimental data on specific material properties.
    Args:
        material_name: Name of the material
        property_type: Type of property (electronic, magnetic, thermal, etc.)
    Returns:
        JSON string with experimental data
    """
    # Mock experimental data lookup
    experimental_data = {}

    return experimental_data


@tool("method_comparison_tool")
def method_comparison_tool(material_class: str, calculation_type: str) -> str:
    """
    Compare different computational methods for specific material classes.
    Args:
        material_class: Class of materials (half-heusler, perovskite, etc.)
        calculation_type: Type of calculation (band_structure, phonon, etc.)
    Returns:
        JSON string with method comparison
    """
    # Mock method comparison data
    comparison_data = {
        "material_class": material_class,
        "calculation_type": calculation_type,
        "method_comparison": [
            {
                "method": "PBE",
                "accuracy": "Good for structural properties",
                "limitations": "Underestimates band gaps",
                "recommended_use": "Initial screening, geometry optimization",
            },
            {
                "method": "HSE06",
                "accuracy": "Excellent for electronic properties",
                "limitations": "Computationally expensive",
                "recommended_use": "Accurate band structures, DOS",
            },
            {
                "method": "PBE+SOC",
                "accuracy": "Essential for heavy elements",
                "limitations": "Increased computational cost",
                "recommended_use": "Topological properties, magnetic systems",
            },
        ],
        "recommendations": {
            "primary_method": "HSE06 for electronic properties",
            "validation_method": "Compare with experimental data",
            "convergence_tests": "Essential for all methods",
        },
    }

    import json

    return json.dumps(comparison_data, indent=2)


# Research tools list
research_tools = [
    literature_search_tool,
    experimental_data_search_tool,
    method_comparison_tool,
]


def wrap_research_model(model: BaseChatModel) -> RunnableSerializable:
    """Model wrapper for research agent"""
    model = model.bind_tools(research_tools)

    def prep_research_state(state: ResearchState):
        material_name = state.get("material_name", "Unknown")
        research_topics = state.get("research_topics", [])
        computational_method = state.get("computational_method", "DFT")

        instructions = RESEARCH_INSTRUCTIONS.format(
            material_name=material_name,
            research_topics=", ".join(research_topics)
            if research_topics
            else "General analysis",
            computational_method=computational_method,
        )

        return [SystemMessage(content=instructions)] + state["messages"]

    return RunnableLambda(prep_research_state, name="ResearchStatePrep") | model


async def research_node(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Main research node for literature analysis"""
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    lm = get_model(model_name)
    runner = wrap_research_model(lm)

    response = await runner.ainvoke(state, config)

    updates = {"messages": [response]}

    # Track tool calls for research progress
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        updates["current_research_step"] = f"executing_{tool_names[0]}"
    else:
        updates["research_complete"] = True
        updates["current_research_step"] = "analysis_complete"

    return updates


def literature_analyzer(state: ResearchState) -> ResearchState:
    """Analyze collected literature for insights and gaps"""
    literature_results = state.get("literature_results", [])

    # Analyze research gaps (simplified)
    research_gaps = []
    methodological_recommendations = []

    # Check for common gaps
    has_soc_studies = any(
        "spin-orbit" in str(result.get("key_findings", [])).lower()
        for result in literature_results
    )

    if not has_soc_studies:
        research_gaps.append("Lack of spin-orbit coupling studies")
        methodological_recommendations.append(
            "Include SOC in calculations for heavy elements"
        )

    has_experimental_validation = any(
        result.get("experimental_data") for result in literature_results
    )

    if not has_experimental_validation:
        research_gaps.append("Limited experimental validation data")
        methodological_recommendations.append(
            "Compare results with available experimental data"
        )

    return {
        "research_gaps": research_gaps,
        "methodological_recommendations": methodological_recommendations,
        "current_research_step": "gaps_analyzed",
    }


def validation_criteria_generator(state: ResearchState) -> ResearchState:
    """Generate validation criteria based on literature analysis"""
    material_name = state.get("material_name", "").lower()

    # Generate validation criteria based on material type
    validation_criteria = {}

    if "heusler" in material_name or "scnisb" in material_name:
        validation_criteria = {
            "band_gap_accuracy": "Within 0.05 eV of experimental value",
            "lattice_parameter": "Within 1% of experimental value",
            "formation_energy": "Within 0.1 eV/atom of experimental value",
            "topological_invariant": "Consistent with experimental topological classification",
            "transport_properties": "Seebeck coefficient within 20% of experimental value",
        }
    else:
        validation_criteria = {
            "band_gap_accuracy": "Within 0.1 eV of experimental value",
            "structural_properties": "Within 2% of experimental values",
            "energetic_properties": "Within 0.2 eV/atom of experimental values",
        }

    return {
        "validation_criteria": validation_criteria,
        "current_research_step": "validation_criteria_set",
    }


def should_continue_research(state: ResearchState) -> str:
    """Determine if research should continue"""
    if state.get("research_complete", False):
        return END

    current_step = state.get("current_research_step", "start")

    if current_step == "start":
        return "research"
    elif current_step.startswith("executing_"):
        return "research"  # Continue with tool execution
    elif current_step == "analysis_complete":
        return "literature_analyzer"
    elif current_step == "gaps_analyzed":
        return "validation_criteria_generator"
    else:
        return END


# Build the research workflow
def build_research_workflow() -> StateGraph:
    """Build the research agent workflow"""
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("literature_analyzer", literature_analyzer)
    workflow.add_node("validation_criteria_generator", validation_criteria_generator)

    # Tool node for research tools
    from langgraph.prebuilt import ToolNode

    workflow.add_node("tools", ToolNode(research_tools))

    # Set entry point
    workflow.set_entry_point("research")

    # Add edges
    workflow.add_conditional_edges("research", should_continue_research)
    workflow.add_conditional_edges("literature_analyzer", should_continue_research)
    workflow.add_conditional_edges(
        "validation_criteria_generator", should_continue_research
    )

    # Tools always return to research node
    workflow.add_edge("tools", "research")

    return workflow


def create_research_agent():
    """Factory function to create the research agent"""
    workflow = build_research_workflow()
    return workflow.compile()


# Export the agent
research_agent = create_research_agent()
