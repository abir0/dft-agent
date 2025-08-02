from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.dft_tools import (
    bands_calc_tool,
    bilbao_crystal_tool,
    convergence_test_tool,
    dos_calc_tool,
    generate_structure_tool,
    materials_project_lookup,
    optimize_geometry_tool,
    pdos_calc_tool,
    qe_input_generator_tool,
    qe_to_ase_tool,
    seekpath_tool,
    submit_job_tool,
)
from agents.llm import get_model, settings

tools = [
    materials_project_lookup,
    generate_structure_tool,
    convergence_test_tool,
    optimize_geometry_tool,
    bands_calc_tool,
    dos_calc_tool,
    pdos_calc_tool,
    qe_to_ase_tool,
    submit_job_tool,
    seekpath_tool,
    bilbao_crystal_tool,
    qe_input_generator_tool,
]


# Agent State
class DFTAgentState(MessagesState, total=False):
    """State holds the chat history and tool calls."""


# System Prompt
instructions = """
You are a world-class DFT expert agent. You have access to tools for:
- Materials Project lookup
- Crystal structure generation & retrieval (Bilbao, SeeK-path)
- QE input generation & QE→ASE conversion
- DFT convergence testing, geometry optimization
- Parallel DFT jobs: band structure, DOS, pDOS
- Job submission (local, SSH, SLURM)

**Workflow Steps**
1) Retrieve or generate structure
2) Test SCF convergence
3) Optimize geometry
4) In parallel: run bands, DOS, pDOS
5) Convert QE inputs ↔ ASE
6) Submit & monitor jobs
7) Collate and report results

Follow these steps automatically.
Use the provided tools. Do not reveal tool responses.
If a tool errors, catch it, fix, and retry up to 3 times.
"""


# Model Wrapper
def wrap_model(model: BaseChatModel) -> RunnableSerializable[DFTAgentState, AIMessage]:
    model = model.bind_tools(tools)
    pre = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StatePrep",
    )
    return pre | model


async def acall_model(state: DFTAgentState, config: RunnableConfig) -> DFTAgentState:
    lm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    runner = wrap_model(lm)
    resp = await runner.ainvoke(state, config)
    return {"messages": [resp]}


# Error Handler
def handle_tool_error(state) -> dict:
    err = state.get("error")
    calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Tool error: {err!r}. Please adjust and retry.",
                tool_call_id=tc["id"],
            )
            for tc in calls
        ]
    }


def make_tool_node(tools_list):
    return ToolNode(tools_list).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Build the Graph
workflow = StateGraph(DFTAgentState)

# 1. Planner / LLM node
workflow.add_node("planner", acall_model)

# 2. Tool nodes for each step
workflow.add_node("mp_lookup", make_tool_node([materials_project_lookup]))
workflow.add_node(
    "structure_gen",
    make_tool_node([generate_structure_tool, bilbao_crystal_tool, seekpath_tool]),
)
workflow.add_node("scf_converge", make_tool_node([convergence_test_tool]))
workflow.add_node("geom_opt", make_tool_node([optimize_geometry_tool]))
# Parallel DFT calculators
workflow.add_node("bands", make_tool_node([bands_calc_tool]))
workflow.add_node("dos", make_tool_node([dos_calc_tool]))
workflow.add_node("pdos", make_tool_node([pdos_calc_tool]))
workflow.add_node("qe2ase", make_tool_node([qe_to_ase_tool]))
workflow.add_node("submit_job", make_tool_node([submit_job_tool]))
# Fallback to planner if needed
# 3. End
workflow.add_node("end", lambda state: {"messages": state["messages"]})

# Edges & Conditions
workflow.set_entry_point("planner")

# After planner decides, route to appropriate tool nodes:
workflow.add_conditional_edges("planner", tools_condition)

# Chain the DFT workflow order:
workflow.add_edge("mp_lookup", "structure_gen")
workflow.add_edge("structure_gen", "scf_converge")
workflow.add_edge("scf_converge", "geom_opt")

# Run bands, dos, pdos in parallel:
workflow.add_edge("geom_opt", "bands")
workflow.add_edge("geom_opt", "dos")
workflow.add_edge("geom_opt", "pdos")

# After all three complete, convert & submit:
workflow.add_edge("bands", "qe2ase")
workflow.add_edge("dos", "qe2ase")
workflow.add_edge("pdos", "qe2ase")
workflow.add_edge("qe2ase", "submit_job")

# Finally back to planner or end:
workflow.add_edge("submit_job", "planner")
# If planner decides to finish:
workflow.add_edge("planner", "end")

# Compile Agent
dft_agent = workflow.compile(
    checkpointer=MemorySaver(),
)
