from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.library.chatbot import chatbot
from agents.library.dft_agent import dft_agent
from core import AgentInfo

DEFAULT_AGENT = "dft-agent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot", graph=chatbot),
    "dft-agent": Agent(description="DFT Agent", graph=dft_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]
