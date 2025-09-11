# CLAUDE.md

Guidance for Claude Code when working in this repository. Keep changes minimal, aligned with existing architecture, and respect enumerated model/provider abstractions.

## Project Overview

DFT-Agent is an LLM‑powered agent system for computational materials science and Density Functional Theory (DFT) workflows. It uses LangGraph for multi-node agent graphs (chat routing → DFT agent → tool execution) and exposes a FastAPI service plus a Streamlit UI. The DFT agent is fully implemented with a rich tool registry (structure generation, Quantum ESPRESSO setup, convergence tests, job submission, Materials Project queries, calculation database management) and a lightweight planning primitive.

## Key Commands

### Development Setup

```bash
# Install dependencies using uv (preferred)
uv sync

# Or using pip
pip install -e .
```

### Running the Application

```bash
# Start both API service and Streamlit frontend
./scripts/run.sh

# Or run components individually:
# API Service
python backend/run_service.py

# Streamlit Frontend  
streamlit run frontend/app.py

# Stop all services
./scripts/stop.sh
```

### Code Quality

```bash
# Run linting with ruff
ruff check backend/
ruff format backend/

# Run tests (pytest is installed as dev dependency)
pytest
```

### LangGraph Development

```bash
# Deploy agent using LangGraph CLI
langgraph up

# Test agent locally
python backend/run_agent.py
```

## Architecture Snapshot

### Backend Highlights

* `backend/agents/agent_manager.py` – Registers compiled graphs; default = `dft_agent`.
* `backend/agents/library/chatbot.py` – General assistant (web search, calc, python, literature tools, optional Asta MCP scholarly tools).
* `backend/agents/library/dft_agent/` – DFT agent graph (`agent.py`), planning primitives (`plan.py`), contextual helpers, tool registry (`tool_registry.py`).
* `backend/agents/dft_tools/` – Tool implementations (structure, Quantum ESPRESSO, convergence tests, Materials Project, database, execution).
* `backend/api/main.py` – FastAPI app; mounts `/agent` router; sets up shared SQLite checkpointing via `AsyncSqliteSaver`.
* `backend/api/endpoints/agent.py` – Core endpoints: `/agent/invoke`, `/agent/stream`, `/agent/feedback`, `/agent/history`.
* `backend/core/models.py` + `backend/agents/llm.py` – Model enums + provider abstraction. Never hardcode raw model strings elsewhere.
* `backend/utils/workspace.py` – Thread‑scoped workspace directories under `WORKSPACE/<thread_id>/`.

### Frontend

* `frontend/app.py` – Streamlit UI consuming the API (invoke/stream, model selection, thread continuity).

### State & Persistence

* Conversation/tool state checkpointed (SQLite `checkpoints.db`).
* File artifacts written into per-thread workspace subfolders (structures/, calculations/, results/, databases/, etc.). Tools return concise summaries; large outputs go to disk.

### Configuration & Auth

* Environment variables via `.env` (see summary below).
* Optional bearer auth: `AUTH_SECRET`. If set, all `/agent/*` requests require `Authorization: Bearer <secret>`.

## LLM / Model Configuration

Enumerated in `backend/core/models.py`; mapping performed in `backend/agents/llm.py` with `_MODEL_TABLE`.

Supported enums (superset; availability depends on provided keys):

* OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-5`
* Groq: `groq-llama-3.1-8b`, `groq-llama-3.3-70b`, `groq-llama-guard-3-8b`
* HuggingFace: `deepseek-r1`, `deepseek-v3`
* Ollama: `ollama` (placeholder alias for user-provided model)
* Fake: deterministic `fake` model for tests (set `USE_FAKE_MODEL=true`)

Add a model: extend enum → update `_MODEL_TABLE` → ensure provider key exists.

## DFT Agent Overview

Primary capabilities implemented in `backend/agents/library/dft_agent/agent.py`:

* Multi-route graph: initial chat node, DFT agent node, custom tool node with thread_id injection.
* Planning system: simple sequential `Plan` + `PlanStep` objects (create/execute/modify). User triggers by natural language (e.g. "create plan for adsorption energy on Pt(111)").
* State fields: working_directory, thread_id, current_structures (capped), last_calculation, workflow type & step, current_plan.
* Tool binding via registry (LangChain tool objects) with custom execution node.

### Tool Registry

Located at `backend/agents/library/dft_agent/tool_registry.py`. Categories:

* Structure: `generate_bulk`, `create_supercell`, `generate_slab`, `add_adsorbate`, `add_vacuum`
* Quantum ESPRESSO: `generate_qe_input`, `submit_local_job`, `check_job_status`, `read_output_file`, `extract_energy`
* Convergence: `kpoint_convergence_test`, `cutoff_convergence_test`, `slab_thickness_convergence`, `vacuum_convergence_test`
* Materials / Analysis: `search_materials_project`, `analyze_crystal_structure`, `find_pseudopotentials`, `calculate_formation_energy`
* Database & Results: `create_calculations_database`, `store_calculation`, `update_calculation_status`, `store_adsorption_energy`, `query_calculations`, `export_results`, `search_similar_calculations`

Add a new tool (summary):

1. Implement function in an appropriate `backend/agents/dft_tools/*.py` module with `@tool` decorator.
2. Keep it pure / deterministic if possible; write large outputs to a workspace file (use `_thread_id` + helpers like `get_subdir_path`).
3. Return a concise JSON-serializable summary (string or dict). Avoid embedding huge raw text.
4. Register in `tool_registry.py` (add to `TOOL_REGISTRY`).
5. (Optional) Update README and planning heuristics if user-facing.

### Workspace Layout

Per-thread root: `WORKSPACE/<thread_id>/` with subfolders:
`structures/ calculations/ convergence_tests/ results/ databases/ kpoints/ kpaths/ optimized/ relaxed/`.
Tools self-manage placement; rely on helper functions in `backend/utils/workspace.py`.

### API Endpoints (Current)

* `POST /agent/invoke` – single execution (returns final AI message wrapper `{messages:[...]}` → last converted to `ChatMessage`).
* `POST /agent/stream` – SSE stream: emits `token` and `message` events; ends with `[DONE]`.
* `POST /agent/history` – retrieve prior messages for a `thread_id`.
* `POST /agent/feedback` – record LangSmith feedback.
* `GET /health` / `GET /info` – health & metadata.

Provide `thread_id` to persist; omit to create a new one. Include bearer token if `AUTH_SECRET` set.

### Planning Commands (User-Facing Conventions)

* "create plan for some goal"
* "show plan"
* "execute plan" / "execute next step"
* "modify plan" (agent currently interprets modifications heuristically)

### Safety / Boundaries

* `python_repl` permits code execution – restrict usage for untrusted users.
* All file operations must remain within the workspace root – avoid raw user-provided absolute paths.
* Large data (trajectory, full logs) → write to file, return path summary.

## Environment Variables (Key)

| Name | Purpose |
|------|---------|
| OPENAI_API_KEY | Enable OpenAI models |
| GROQ_API_KEY | Enable Groq models |
| HF_API_KEY | HuggingFace endpoint models |
| OLLAMA_MODEL / OLLAMA_BASE_URL | Local Ollama inference |
| AUTH_SECRET | Optional API bearer auth |
| MP_API_KEY | Materials Project queries |
| ASTA_KEY | Scholarly (Asta MCP) tools |
| DEFAULT_MODEL | Override auto-selected default |
| USE_FAKE_MODEL | Allow fake deterministic model |
| MODE=dev | Enable reload/dev behaviors |

At least one provider key (or `USE_FAKE_MODEL=true`) is required; else startup raises.

## Contribution Guidelines (Claude-Specific)

* Before large refactors, search for existing helper patterns (e.g. how tools inject `_thread_id`).
* Do not rename graph node labels (`chat`, `dft_agent`, `tools`) without updating routing logic and docs.
* When adding new external dependencies, update `pyproject.toml` and prefer well-maintained libs.
* Keep agent state minimal. Persist large artifacts to the filesystem.
* Update both README and this file when expanding core capabilities (new tool categories, new agents, new endpoint patterns).
* Run Ruff & (if present) tests before concluding edits.

## Quick Quality Checklist (Before PR)

1. `ruff check backend/` passes (or intentional ignores documented).
2. New tools appear in `TOOL_REGISTRY` and import cleanly.
3. No hardcoded model strings outside `llm.py` or enums.
4. Workspace file writes confined to thread directory.
5. Streaming behavior unchanged (event keys: `token`, `message`).
6. README/CLAUDE.md updated if public-facing behavior changed.

## Future Enhancements (Reference)

* Additional engines (VASP, Gaussian) as separate tool modules.
* Remote scheduler integration (SLURM/PBS job submission tools).
* Advanced multi-branch planning with dependency graphs.
* Visualization endpoints (band structure, convergence plots served via API or UI module).
* Provenance metadata export + reproducibility manifests.

---

If unsure about a change, prefer adding a small utility or doc note rather than refactoring core agent logic. Keep edits surgical and reversible.
