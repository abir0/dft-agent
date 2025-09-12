from dataclasses import dataclass
from typing import Callable, Optional

from langgraph.graph.state import CompiledStateGraph

from backend.core import AgentInfo

DEFAULT_AGENT = "chatbot"


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

    return chatbot


def _create_dft_agent():
    from backend.agents.library.dft_agent.agent import dft_agent

    return dft_agent


def _create_unified_agent():
    from backend.agents.library.unified_agent import unified_agent

    return unified_agent


def _create_slurm_scheduler():
    from backend.agents.library.slurm_scheduler.agent import slurm_scheduler_agent

    return slurm_scheduler_agent


# Supervisor agent removed - functionality consolidated into chatbot


agent_configs: dict[str, AgentConfig] = {
    "chatbot": AgentConfig(
        description="Main DFT Agent - Comprehensive assistant for DFT workflows, structure generation, QE input creation, SLURM job management, and materials science calculations",
        factory=_create_chatbot,
    ),
    "dft_agent": AgentConfig(
        description="Expert DFT agent for computational materials science workflows",
        factory=_create_dft_agent,
    ),
    "unified_agent": AgentConfig(
        description="Unified agent combining general and computational capabilities with intelligent planning",
        factory=_create_unified_agent,
    ),
    "slurm_scheduler": AgentConfig(
        description="SLURM job scheduler agent for HPC job management and automation",
        factory=_create_slurm_scheduler,
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agent_configs[agent_id].get_graph()


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=config.description)
        for agent_id, config in agent_configs.items()
    ]
