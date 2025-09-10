import asyncio
from datetime import datetime
from pathlib import Path

from ddgs import DDGS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.agents.llm import get_model, settings
from backend.agents.tools import calculator, python_repl
from backend.agents.asta_mcp_client import get_specific_asta_tools


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# Create web search tool using ddgs
@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this tool to find current information or answer questions that require up-to-date data."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No results found for query: {query}"

            formatted_results = []
            for result in results:
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                href = result.get('href', 'No link')
                formatted_results.append(f"**{title}**\n{body}\n{href}\n")

            return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching for '{query}': {str(e)}"

# Initialize base tools
base_tools = [web_search, calculator, python_repl]

# Global variable to cache loaded tools
_all_tools = None
_asta_tools_loaded = False

async def get_all_tools():
    """Get all tools including Asta MCP tools with proper async handling."""
    global _all_tools, _asta_tools_loaded
    
    if _all_tools is not None:
        return _all_tools
    
    # Start with base tools
    _all_tools = base_tools.copy()
    
    # Try to load Asta MCP tools
    if not _asta_tools_loaded:
        try:
            asta_tool_names = [
                "search_papers_by_relevance",
                "search_paper_by_title", 
                "get_papers",
                "get_citations",
                "search_authors_by_name",
                "get_author_papers"
            ]
            asta_tools = await get_specific_asta_tools(asta_tool_names)
            _all_tools.extend(asta_tools)
            _asta_tools_loaded = True
            print(f"Loaded {len(asta_tools)} Asta MCP tools")
        except Exception as e:
            print(f"Warning: Could not load Asta MCP tools: {e}")
            _asta_tools_loaded = True  # Mark as attempted to avoid retrying
    
    return _all_tools

# For now, start with base tools only
# Asta tools will be loaded lazily when the agent runs
tools = base_tools

# System message
current_date = datetime.now().strftime("%B %d, %Y")
images_dir = f"{settings.ROOT_PATH}/data/images"
Path(images_dir).mkdir(parents=True, exist_ok=True)
datasets_dir = f"{settings.ROOT_PATH}/data/raw_data"

instructions = f"""
    You are a helpful chat assistant with expertise in materials science and DFT calculations,
    with the ability to search the web, search scientific literature, and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - When searching, be persistent. Expand your query bounds if the first search returns no results.
    - If a search comes up empty, expand your search before giving up.
    - Use the search_papers_by_relevance tool to find peer-reviewed scientific literature by keyword.
    - Use the search_paper_by_title tool to find specific papers by their title.
    - Use the get_papers tool to get detailed information about specific papers using their ID.
    - Use the get_citations tool to find papers that cite a specific paper.
    - Use the search_authors_by_name and get_author_papers tools to find papers by specific researchers.
    - If Asta scientific paper search tools return 403 Forbidden errors, inform the user that there may be an API key permission issue and suggest using web search instead.
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format or markdown format.
    - Use Python REPL tool for data analysis and visualization tasks.
    - If Python REPL shows error fix the error in code and run again, if you are failing more than 3 times give up.
    - For data processing and analysis, use pandas library.
    - Local datasets are available at: {datasets_dir}
    - Load datasets with absolute paths via pandas, e.g., `import pandas as pd; from pathlib import Path; df = pd.read_csv(Path("{datasets_dir}") / "file.csv")`. Prefer read-only access; do not move or rename source files.
    - For charts generation, use seaborn or matplotlib.
    - ALWAYS save the plots/charts into the following folder: {images_dir}
    - Only include markdown-formatted links to citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - When displaying image to the user, use html <img> tag, instead of markdown.
    ALWAYS USE IMG TAG FOR LINKING IMAGES.
    """


async def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    all_tools = await get_all_tools()
    model = model.bind_tools(all_tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = await wrap_model(m)
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


async def create_tool_node_with_fallback() -> dict:
    all_tools = await get_all_tools()
    return ToolNode(all_tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


async def create_chatbot_workflow():
    """Create the chatbot workflow with all tools loaded."""
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Define nodes
    workflow.add_node("chatbot", acall_model)
    workflow.add_node("tools", await create_tool_node_with_fallback())
    
    # Define edges
    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    workflow.add_edge("tools", "chatbot")
    
    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())

# Create a synchronous wrapper that initializes the workflow lazily
class LazyWorkflow:
    def __init__(self):
        self._workflow = None
    
    async def get_workflow(self):
        if self._workflow is None:
            self._workflow = await create_chatbot_workflow()
        return self._workflow
    
    async def ainvoke(self, *args, **kwargs):
        workflow = await self.get_workflow()
        return await workflow.ainvoke(*args, **kwargs)
    
    async def astream(self, *args, **kwargs):
        workflow = await self.get_workflow()
        async for event in workflow.astream(*args, **kwargs):
            yield event
    
    async def astream_events(self, *args, **kwargs):
        workflow = await self.get_workflow()
        async for event in workflow.astream_events(*args, **kwargs):
            yield event
    
    def __getattr__(self, name):
        # For any other methods, delegate to the workflow once it's created
        def method_wrapper(*args, **kwargs):
            async def async_method():
                workflow = await self.get_workflow()
                method = getattr(workflow, name)
                if asyncio.iscoroutinefunction(method):
                    return await method(*args, **kwargs)
                else:
                    return method(*args, **kwargs)
            return async_method()
        return method_wrapper

# Create the lazy workflow instance
chatbot = LazyWorkflow()
