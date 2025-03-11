from pathlib import Path

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.llm import get_model, settings
from agents.tools import postgres_db_search, python_repl


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# Add the tools
web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [postgres_db_search, python_repl, web_search]

# Image folder path
images_dir = f"{settings.ROOT_PATH}/data/images"
Path(images_dir).mkdir(parents=True, exist_ok=True)

# System message
instructions = f"""   
    You are world class world class data analyst.
    You are authorized to query data from PostgreSQL database.
    You need to first query sample data to understand the schema.
    Then, you can plan on how to analyze the data and what analysis and charts to be generated.
    You have access to Python REPL which you can use to generate code, analyze data, and create charts/plots.
    Step 0: Query the schema of the database to understand the columns and data types.
    Step 1: Based on this schema and user query, create a detailed plan on data analysis and visualization (charts/plots).
    Step 2: Query all the required data based on the plan using db search tool, this will save the data in a file.
    Step 3: Generate Python code to load the data from file, then (1) analyze data and (2) create visualizations.
    Step 4: Run the code using Python REPL and get the analysis results.
    Step 5: Create a final report on the analysis including the image links (<img> tag) for the visualization or charts.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Generate code always based on the user query, your detailed plan, and the data schema.
    - Use the provided tools to search/query for data from the databases using SQL-like query, run code to
    generate charts or graphs or summary statistics, and search web for relevant information if needed.
    - If Python REPL shows error fix the error in code and run again, if you are failing more than 3 times give up.
    - For data processing and analysis, use pandas library.
    - Use the returned filename from database search tool to load data into pandas.
    - For charts generation, use seaborn or matplotlib.
    - ALWAYS save the plots/charts into the following folder: {images_dir}
    - Only include markdown-formatted links to citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - When displaying image to the user, use html <img> tag, instead of markdown.
    ALWAYS USE IMG TAG FOR LINKING IMAGES.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Agent functions
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Define the graph
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("analyst", acall_model)
workflow.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges
workflow.set_entry_point("analyst")
workflow.add_conditional_edges(
    "analyst",
    tools_condition,
)
workflow.add_edge("tools", "analyst")
workflow.add_edge("analyst", END)

# Compile the graph
data_analyst = workflow.compile(
    checkpointer=MemorySaver(),
)
