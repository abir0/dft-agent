# CRYSTAL

## Agentic AI Assistant for Computational Materials Science

Streamlined DFT workflows • HPC automation • Extensible multi‑agent architecture

[Docs](docs/COMPREHENSIVE_GUIDE.md) · [Installation Guide](docs/guides/installation.md) · [SLURM Guide](docs/guides/SLURM_SCHEDULER.md) · [Examples](docs/examples)  
[Open Issues](../../issues) · [Discussions](../../discussions)

## Overview

DFT Agent is a production‑oriented, tool‑rich autonomous assistant for Density Functional Theory (DFT) and broader computational materials research. It combines:

* A multi‑agent graph built with **LangGraph** (chat, domain DFT expert, SLURM scheduler)
* A **FastAPI** backend with streaming, state checkpointing, and structured tool execution
* A **Thread‑scoped workspace** system for deterministic artifact storage
* A **Streamlit frontend** (project name: CRYSTAL) for interactive exploration

The system orchestrates structure generation, Quantum ESPRESSO (QE) input authoring, convergence studies, HPC job lifecycle management, and materials data retrieval—while remaining fully extensible via a clean tool registry pattern.

## Feature Highlights

| Category | Capabilities |
|----------|--------------|
| Multi‑Agent | Chatbot, DFT Agent, SLURM Scheduler (pluggable registry) |
| DFT Workflows | QE input generation, k‑point & cutoff convergence, slab/vacuum tuning, structure transforms |
| Materials Data | Materials Project integration, structure parsing & analysis (pymatgen utilities) |
| HPC Automation | SLURM submission, monitoring, queue inspection, resource specification |
| Persistence | SQLite checkpointing + per‑thread filesystem (idempotent tool outputs) |
| Streaming | Server‑Sent Events (token + message events) |
| Extensibility | Decorated tools with deterministic return shapes & workspace routing |
| Safety | Scoped file IO, optional auth secret, configurable model providers |

## Architecture Snapshot

```text
┌──────────────────────────┐
│ Streamlit Frontend       │  interactive chat
└─────────────▲────────────┘
              │ REST API
┌─────────────┴────────────┐
│ FastAPI Service          │  routing, streaming, feedback
│  • /agent/invoke         │
│  • /agent/stream         │
│  • /agent/history        │
└─────────────▲────────────┘
              │ LangGraph compiled graphs
┌─────────────┴────────────┐
│ Agent Graphs             │ research_agent, dft_agent, slurm_agent
│  • Routing / Planning    │
│  • Tool Execution Node   │
└─────────────▲────────────┘
              │ Tool calls
┌─────────────┴────────────┐
│ Tool Registry            │ ASE, QE, MP, SLURM
└─────────────▲────────────┘
              │ Workspace API
┌─────────────┴────────────┐
│ Thread Workspace         │ structured directories + artifacts
└──────────────────────────┘
```

Key modules: `backend/api/main.py`, `backend/agents/library/`, `backend/agents/dft_tools/`, `backend/utils/workspace.py`.

## Quick Start

### Prerequisites

* Python 3.12+
* `uv` (or fallback to `pip`)
* At least one model provider key (OpenAI, Groq, HF, Cerebras, Ollama)

### Install & Run (Local)

```bash
git clone https://github.com/abir0/dft-agent.git
cd dft-agent
uv sync  # installs dependencies from pyproject.toml / uv.lock
cp env.example .env

# Start backend API
uv run python backend/run_service.py

# In a second terminal start the frontend
uv run streamlit run frontend/app.py
```

### Using Docker

```bash
docker-compose up --build
```

**Access:**

* Web UI: <http://localhost:8501>
* API Root: <http://localhost:8083>
* OpenAPI Docs: <http://localhost:8083/docs>

## Environment Configuration

| Variable | Purpose |
|----------|---------|
| MODE | Set to `dev` for hot reload behaviors |
| AUTH_SECRET | Optional bearer token for protected endpoints |
| OPENAI_API_KEY | OpenAI model access |
| CEREBRAS_API_KEY | Cerebras models |
| GROQ_API_KEY | Groq LLaMA models |
| HF_API_KEY | HuggingFace Inference / endpoints |
| OLLAMA_MODEL | Local Ollama model name (e.g. `llama3`) |
| OLLAMA_BASE_URL | Ollama server URL |
| USE_FAKE_MODEL | Set `true` to enable deterministic stub model |
| DEFAULT_MODEL | Override auto-selected default model |
| MP_API_KEY | Materials Project queries |
| ASTA_KEY | Scholarly literature / Asta MCP tools |
| DATABASE_URL | Optional external DB (currently unused placeholder) |

If no real API keys are provided and `USE_FAKE_MODEL` is not true, startup will raise an error.

## Agents

| Agent | Key | Description |
|-------|-----|-------------|
| Chatbot | `chatbot` | General assistant (search, calculator, Python REPL, literature) |
| DFT Agent | `dft_agent` | Domain workflow planner + materials + QE + databases |
| SLURM Scheduler | `slurm_scheduler` | HPC job submission, queue mgmt, monitoring |

Switch agents via frontend settings or by passing `agent` param to the API.

## Tooling Domains

* Structure manipulation & generation (`structure_tools.py`)
* Quantum ESPRESSO input & execution helpers (`qe_tools.py`)
* Convergence & parameter studies (`convergence_tools.py`)
* Materials Project search / retrieval (`pymatgen_tools.py`, MP API)
* SLURM job lifecycle (`slurm_tools.py`)
* Local database & result curation (`database_tools.py`)

Large outputs are stored under the active thread workspace: `WORKSPACE/<thread_id>/...`.

### Workspace Layout

```text
WORKSPACE/<thread_id>/
calculations/ 
databases/
results/
structures/
pseudos/
...
```

## API Usage

`POST /agent/invoke`

```json
{
    "input": "Generate a QE input for fcc Cu and run a cutoff convergence study",
    "thread_id": "<uuid>",
    "agent": "dft_agent"
}
```

Streaming: `GET /agent/stream?thread_id=<uuid>&agent=dft_agent&input=...` yields `token` and `message` events followed by `[DONE]`.

## Frontend (CRYSTAL)

The Streamlit UI provides:

* Real‑time streaming transcripts
* Agent + model selector
* New chat / shareable thread IDs
* Basic privacy notice & feedback hook

Launch (after backend):

```bash
uv run streamlit run frontend/app.py
```

## Development

### Code Style & Tooling

* Python project managed by `uv`
* Ruff / type checking recommended (configure locally)
* Modular tool registration for easy extension

### Add a New Tool (Summary)

1. Create function in `backend/agents/dft_tools/<new_tool>.py` with `@tool` decorator.
2. Persist large artifacts inside the active workspace using provided helpers.
3. Import and register in the central registry (see `tool_registry.py`).
4. Add a lightweight test verifying registry presence.

### Tests

```bash
uv run pytest -q
```

## Roadmap

* Additional DFT engines (VASP / CASTEP adapters)
* Advanced multi-step automatic planners
* Result visualization panels (band structures, DOS)
* Caching & reuse of convergence curves
* Expanded materials databases integration

## Contributing

We welcome focused, well-scoped contributions:

1. Open an issue describing the change.
2. Fork & branch: `feature/<feature-name>`.
3. Add/update tests & docs for user-visible behavior.
4. Submit PR referencing the issue; keep commits atomic.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).

## Acknowledgments

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [pymatgen](https://pymatgen.org/)
* [Materials Project](https://materialsproject.org/)
* [ASE](https://wiki.fysik.dtu.dk/ase/)
* [Quantum ESPRESSO](https://www.quantum-espresso.org/)
* Broader computational materials science community

---

Have ideas? Open an issue or start a discussion.  
**Happy computing ⚛️**
