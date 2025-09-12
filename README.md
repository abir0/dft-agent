# DFT Agent

An intelligent, tool-rich agent system for autonomous materials research and Density Functional Theory (DFT) workflows. Built with LangGraph (graph-based agent orchestration) and FastAPI, featuring persistent conversation state, structured workspaces, and an extensible tool registry for real computational tasks (structure generation, QE input creation, convergence testing, job submission, database storage, etc.).

## 🚀 Project Status

The DFT agent is now implemented with a comprehensive tool suite and planning capabilities.

Currently available:

- ✅ LangGraph multi-node agent graphs (chat routing → DFT agent → tools)
- ✅ REST API with invoke + streaming endpoints (`/agent/*`)
- ✅ Persistent threaded state via SQLite checkpointing (thread_id based)
- ✅ Multi-model LLM abstraction (OpenAI, Groq, HuggingFace, Ollama, Fake)
- ✅ General Chatbot agent (web search, calculator, REPL, literature tools)
- ✅ Full DFT Agent with workflow planning & execution
- ✅ Quantum ESPRESSO input generation & local job submission utilities
- ✅ Convergence testing (k-point, cutoff, slab thickness, vacuum)
- ✅ Materials Project integration (structure search, analysis, formation energy)
- ✅ Structure manipulation (bulk/slab generation, supercells, adsorption, vacuum)
- ✅ Result/database tooling (store/query/export/search similar calculations)
- ✅ Streaming SSE with token + structured message events
- ✅ Authentication via bearer token (optional)
- ✅ Streamlit UI
- ⏳ Additional engine interfaces (VASP, Gaussian) – extensible
- ⏳ Advanced workflow orchestration & visualization enhancements

## 📁 Project Structure

```text
dft-agent/
├── backend/                    # Core backend services
│   ├── agents/                 # Agent implementations
│   │   ├── library/           # Agent library (chatbot + dft_agent graph)
│   │   │   ├── chatbot.py     # General-purpose chatbot agent
│   │   │   └── dft_agent/     # DFT agent implementation (graph, planning, registry)
│   │   ├── dft_tools/         # DFT tool implementations (structure/QE/db/convergence)
│   │   ├── agent_manager.py   # Agent registry and management
│   │   ├── client.py          # Agent client interface
│   │   ├── llm.py            # LLM model configurations
│   │   └── tools.py          # General-purpose tools
│   ├── api/                   # FastAPI application
│   │   ├── endpoints/         # API route handlers
│   │   ├── main.py           # FastAPI app initialization
│   │   └── dependencies.py   # Dependency injection
│   ├── core/                  # Core data models and schemas
│   │   ├── models.py         # Pydantic models
│   │   ├── schema.py         # API schemas
│   │   └── exceptions.py     # Custom exceptions
│   ├── utils/                 # Utility modules
│   └── web_browser/          # Web browsing capabilities
├── frontend/                  # Streamlit web interface
│   └── app.py                # Main Streamlit application
├── data/                     # Data storage
│   └── raw_data/             # Raw datasets (adsorption data)
├── scripts/                  # Deployment and utility scripts
└── logs/                     # Application logs
```

## 🛠 Technology Stack

### Core Framework

- **LangGraph**: Agent orchestration and state management
- **FastAPI**: High-performance API framework
- **Pydantic**: Data validation and serialization
- **SQLite**: Conversation checkpointing and persistence

### LLM Integrations

Model selection is abstracted through enums; never hardcode raw model strings outside `backend/agents/llm.py`.

Supported model families (availability depends on supplied API keys/env):

- OpenAI: gpt-4o, gpt-4o-mini, gpt-5
- Groq: groq-llama-3.1-8b, groq-llama-3.3-70b, groq-llama-guard-3-8b (safety)
- HuggingFace: deepseek-r1, deepseek-v3
- Ollama: configurable local model (default generic alias)
- Fake: deterministic mock for testing

### Materials Science & DFT Libraries

- **ASE**: Atomic structure manipulation / I/O
- **Pymatgen**: Structure analysis & materials data
- **Materials Project API**: Remote materials database search
- **SeekPath**: Brillouin zone path generation (band structures)
- **NumPy / Pandas / Matplotlib / Seaborn**: Analysis & visualization utilities

### Agent Tools & Utilities

- Web search (DuckDuckGo)
- Literature retrieval & open-access full text extraction
- Calculator & Python REPL sandbox
- DFT tool registry (see below)
- Streaming + stateful workspace management

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd dft-agent
   ```

2. **Install dependencies:**

   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Environment setup:**

   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys
   # Required: OPENAI_API_KEY, GROQ_API_KEY, etc.
   ```

### Running the Application

#### Backend API Server

Default host/port are `0.0.0.0:8080` (see `backend/settings.py`).

```bash
# Development (auto-reload if MODE=dev)
cd backend
python run_service.py

# Or explicit uvicorn (matches settings PORT)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

#### Frontend Interface

```bash
# Streamlit app
cd frontend
streamlit run app.py
```

#### Using Docker

```bash
# Build and run with docker-compose
docker-compose up --build
```

### API Usage

All agent interactions are under the `/agent` prefix with optional bearer auth.

Core endpoints:

- `POST /agent/invoke` (or `/agent/{agent_id}/invoke`) – single-turn execution
- `POST /agent/stream` – Server-Sent Events (SSE) streaming
- `POST /agent/feedback` – record LangSmith feedback
- `POST /agent/history` – retrieve conversation history for a thread
- `GET /health` – service health
- `GET /info` – models + agents metadata

SSE events emitted:

- `token` – incremental LLM text (when `stream_tokens=true`)
- `message` – structured message objects (tool outputs, AI replies)
- `[DONE]` – stream termination sentinel

Threaded persistence:

- Provide a `thread_id` in the JSON body to resume context & workspace.
- Omit to auto-generate a new isolated conversation/workspace.

Authentication (optional):

- If `AUTH_SECRET` is set, include `Authorization: Bearer <SECRET>` header.

Example (invoke chatbot):

```python
import httpx, uuid

BASE = "http://localhost:8080"
thread_id = str(uuid.uuid4())
payload = {
    "message": "Give me a brief overview of DFT in 3 bullet points.",
    "model": "gpt-4o-mini",
    "thread_id": thread_id,
}
r = httpx.post(f"{BASE}/agent/invoke", json=payload)
print(r.json())
```

Streaming with async client (DFT agent):

```python
import asyncio, httpx

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", "http://localhost:8080/agent/stream", json={
            "message": "Create a plan to compute the adsorption energy of CO on Pt(111)",
            "model": "gpt-5",
            "thread_id": "example-thread-1"
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    print(line[6:])

asyncio.run(main())
```

Invoking a specific agent:

```python
httpx.post(f"{BASE}/agent/dft_agent/invoke", json={
    "message": "Generate a QE SCF input for fcc Pt (4 atom conventional cell).",
    "model": "gpt-4o",
    "thread_id": thread_id
})
```

## 🔧 Configuration

### Model Configuration

Models are enumerated in `backend/core/models.py`; provider dispatch & instantiation lives in `backend/agents/llm.py`. Add new enum members then map them in `_MODEL_TABLE` – the API surfaces availability via `/info`.

### Agent Configuration

`backend/agents/agent_manager.py` registers compiled graphs. Default agent: `dft_agent`.

## 📊 Current Agents

### Chatbot Agent

General multi-domain assistant with:

- Web search (DuckDuckGo)
- Calculator & Python REPL
- Open-access literature text extraction
- (Lazy) Asta MCP scholarly data tools (papers, citations, authors) when available

### DFT Agent Overview

Expert computational materials assistant that performs planning + tool execution. Key features:

- Plan creation & step execution ("create plan for adsorption energy on Pt(111)")
- Structure generation: bulk, supercell, slab, adsorption, vacuum
- Quantum ESPRESSO: input generation, local job submission, status, output parsing
- Convergence testing: k-point, cutoff, slab thickness, vacuum spacing
- Materials Project: search, structure analysis, formation energy
- Workspace-aware: per-thread directories under `WORKSPACE/<thread_id>/...`
- Lightweight state (recent structures, last calculation, workflow step, plan)
- Database utilities: create/store/update/query/export/search calculations

#### DFT Tool Registry (selected)

Structure: `generate_bulk`, `create_supercell`, `generate_slab`, `add_adsorbate`, `add_vacuum`
QE & Execution: `generate_qe_input`, `submit_local_job`, `check_job_status`, `read_output_file`, `extract_energy`
Convergence: `kpoint_convergence_test`, `cutoff_convergence_test`, `slab_thickness_convergence`, `vacuum_convergence_test`
Materials & Analysis: `search_materials_project`, `analyze_crystal_structure`, `find_pseudopotentials`, `calculate_formation_energy`
Database & Results: `create_calculations_database`, `store_calculation`, `update_calculation_status`, `store_adsorption_energy`, `query_calculations`, `export_results`, `search_similar_calculations`

All tool implementations live in `backend/agents/dft_tools/` and are registered via `tool_registry.py`.

Example DFT workflow (manual steps):

1. Generate bulk: ask "Generate fcc Pt bulk"
2. Create slab: "Create a (1 1 1) slab with 4 layers"
3. Add adsorbate: "Add CO adsorbate"
4. Generate QE input: "Make an scf QE input with 50 Ry cutoff"
5. Submit job: "Submit local QE job" (specify input mapping)
6. Check status & extract energy

Or request a plan: "Create plan for adsorption energy of CO on Pt(111)" then "execute plan".

## 📈 Development Roadmap (Updated)

Phase 1 (Foundation) – Complete

- LangGraph agent graphs, multi-model support, API, Streamlit, base tools

Phase 2 (Core DFT) – Largely Complete

- Structure generation & manipulation
- Materials Project integration
- QE input + submission + parsing
- Convergence tests
- Local calculation database tooling
- Planning system (simple sequential plans)

Phase 3 (Planned Enhancements)

- Additional engines (VASP, Gaussian interfaces)
- Advanced multi-branch workflows & dependency tracking
- Automated job chaining & error recovery loops
- Enhanced visualization (band structures, charge densities)
- HPC / remote scheduler integration (SLURM, PBS)
- Result provenance & reproducibility metadata exporter
- Rich Streamlit UI modules for trajectory & convergence plots

Phase 4 (Stretch)

- Active learning loops (suggest next calculations)
- Uncertainty-aware workflow steering
- Multi-agent collaboration (planner/critic/executor roles)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the excellent [LangGraph](https://github.com/langchain-ai/langgraph) framework
- Utilizes the [Materials Project](https://materialsproject.org/) ecosystem
- Inspired by the computational materials science community

---

### 🔐 Environment Variables (Summary)

| Variable | Purpose |
|----------|---------|
| OPENAI_API_KEY | Enable OpenAI models |
| GROQ_API_KEY | Enable Groq Llama models |
| HF_API_KEY | Enable HuggingFace endpoint models |
| OLLAMA_MODEL / OLLAMA_BASE_URL | Local Ollama model config |
| AUTH_SECRET | Optional bearer token for API protection |
| MP_API_KEY | Materials Project queries |
| ASTA_KEY | Scholarly (Asta MCP) tools |
| DEFAULT_MODEL | Override initial default model |
| MODE=dev | Enables auto-reload & dev behaviors |

At least one model provider key (or USE_FAKE_MODEL=true) must be set; otherwise initialization will fail.

### 🗂 Workspace Layout

Per-thread directories under `WORKSPACE/<thread_id>/` with standardized subfolders:

```text
calculations/  convergence_tests/  databases/  kpaths/  kpoints/  optimized/  relaxed/  results/  structures/
```

Tools automatically place generated artifacts in the correct subfolder (e.g. QE inputs under `calculations/`). Return payloads are concise; large artifacts are written to disk for persistence.

### 🛡 Safety Notes

- The `python_repl` tool executes arbitrary code – restrict or disable in production deployments.
- All file operations are scoped to the workspace root; avoid passing untrusted absolute paths.

### ❓ Support

Open an issue or discussion for feature requests, bug reports, or extension ideas.

Happy computing!
