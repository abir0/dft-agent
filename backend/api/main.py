import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from backend.api.endpoints import agent
from backend.core import ServiceMetadata
from backend.settings import settings

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    # TODO: Create different checkpointer for multiple agents
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        agents = get_all_agent_info()
        for a in agents:
            agent = get_agent(a.key)
            agent.checkpointer = saver
        yield


# FastAPI app initialization
app = FastAPI(
    title=settings.TITLE,
    version=settings.VERSION,
    swagger_ui_parameters={
        "docExpansion": "none",
        "syntaxHighlight.theme": "obsidian",
    },
    lifespan=lifespan,
)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: callable) -> dict:
    """Add processing time of each request to the response headers."""
    start_time = perf_counter()
    response = await call_next(request)
    process_time = perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Default API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


# Register API endpoints
app.include_router(agent.router)
