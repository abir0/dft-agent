# backend/agents/agent_manager.py

from backend.core import AgentInfo

DEFAULT_AGENT = "dft_planner"

class DFTPlannerAgent:
    checkpointer = None  # optional hook

    def plan(self, request_text: str, hints: dict | None = None, code: dict | None = None) -> dict:
        # Import here so a planner bug doesn't crash FastAPI boot
        from backend.agents.library.dft_agent.planner_graph import generate_plan
        return generate_plan(request_text, hints or {}, code or {})

# registry meta for /info
_AGENTS = {
    "dft_planner": AgentInfo(key="dft_planner", name="DFT Planner",
                             description="Plans DFT workflows and returns strict JSON."),
    "chatbot":     AgentInfo(key="chatbot", name="Chatbot",
                             description="A simple chatbot."),
}

def get_agent(agent_id: str):
    if agent_id == "dft_planner":
        return DFTPlannerAgent()
    elif agent_id == "chatbot":
        # import lazily too
        from backend.agents.library.chatbot import chatbot
        return chatbot  # if you still want to return the compiled graph for the chatbot
    raise KeyError(agent_id)

def get_all_agent_info() -> list[AgentInfo]:
    return list(_AGENTS.values())
