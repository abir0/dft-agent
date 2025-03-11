import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

from agents import DEFAULT_AGENT, get_agent  # noqa: E402

agent = get_agent(DEFAULT_AGENT)


async def main() -> None:
    inputs = {"messages": [("user", "Which are the top iPaaS services?")]}
    result = await agent.acall_model(
        inputs,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )
    result["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
