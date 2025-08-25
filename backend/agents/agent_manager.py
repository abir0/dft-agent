from dataclasses import dataclass

from core import AgentInfo
from langgraph.graph.state import CompiledStateGraph

from backend.agents.library.chatbot import chatbot

DEFAULT_AGENT = "dft-agent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot", graph=chatbot),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]
