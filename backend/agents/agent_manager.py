from dataclasses import dataclass
from typing import Callable, Optional

from langgraph.graph.state import CompiledStateGraph
from backend.core import AgentInfo

DEFAULT_AGENT = "dft_agent"


@dataclass
class AgentConfig:
    description: str
    factory: Callable[[], CompiledStateGraph]
    _cached_graph: Optional[CompiledStateGraph] = None

    def get_graph(self) -> CompiledStateGraph:
        """Lazy-load the graph when first accessed."""
        if self._cached_graph is None:
            self._cached_graph = self.factory()
        return self._cached_graph


def _create_chatbot():
    from backend.agents.library.chatbot import chatbot
    return chatbot()


def _create_dft_agent():
    from backend.agents.library.dft_agent.agent import dft_agent
    return dft_agent


agent_configs: dict[str, AgentConfig] = {
    "chatbot": AgentConfig(
        description="A simple chatbot", 
        factory=_create_chatbot
    ),
    "dft_agent": AgentConfig(
        description="Expert DFT agent for computational materials science workflows",
        factory=_create_dft_agent,
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agent_configs[agent_id].get_graph()


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=config.description)
        for agent_id, config in agent_configs.items()
    ]
