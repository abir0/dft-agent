import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

load_dotenv()

from backend.agents import DEFAULT_AGENT, get_agent  # noqa: E402

agent = get_agent(DEFAULT_AGENT)


async def main() -> None:
    inputs = {"messages": [HumanMessage(content="Which are the top iPaaS services?")]}
    result = await agent.ainvoke(
        inputs,
        config=RunnableConfig(configurable={"thread_id": str(uuid4())}),
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
