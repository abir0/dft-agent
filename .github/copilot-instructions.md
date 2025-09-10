# Copilot Project Instructions

Concise, actionable guidance for AI coding agents working in this repo. Focus on preserving existing architecture, following established patterns, and avoiding unsafe changes.

## 1. Architecture Snapshot
- Backend service: FastAPI (`backend/api/main.py`) exposing agent endpoints under `/agent/*` plus `/health` and `/info`.
- Agents built with LangGraph (`backend/agents/`):
  - `library/chatbot.py`: General-purpose conversational agent with web search, calculator, python REPL tools.
  - `library/dft_agent/agent.py`: Specialized DFT workflow agent with custom routing (chat vs DFT vs tools) and a large domain prompt.
  - Tool registry: `library/dft_agent/tool_registry.py` + domain DFT tools in `backend/agents/dft_tools/*.py`.
- Agent registry: `backend/agents/agent_manager.py` maps agent keys to compiled graphs.
- LLM provider abstraction: `backend/agents/llm.py` + model enums in `backend/core/models.py`; dynamic selection via `settings.DEFAULT_MODEL`.
- State checkpointing: SQLite (`checkpoints.db`) via `AsyncSqliteSaver` configured in `api/main.py` lifespan.
- Workspace & per-thread file organization: `backend/utils/workspace.py`; thread/session ID embedded in LangGraph config (`configurable.thread_id`).
- Schemas & API data contracts: `backend/core/schema.py`.

## 2. Key Conventions & Patterns
- Thread/Conversation continuity: Always pass `thread_id` (or let API create one). Workspace dirs live under `WORKSPACE/<thread_id>/` with standardized subfolders.
- Tools: Added by binding via `model.bind_tools([...])`; DFT agent manually processes tool calls in `custom_tool_node` to inject `thread_id`.
- Prompts: Large system instructions embedded inline; if refactoring, extract to a module but keep semantic content stable.
- Models: Use enums (`OpenAIModelName`, `GroqModelName`, etc.)—never hardcode raw model strings outside `llm.py` or enums.
- Settings: Centralized in `backend/settings.py`; do not mutate global `settings` except via environment. Use `settings.is_dev()` for dev-only behavior.
- Return shape for invoke endpoints: `{"messages": [...]}` from graph; API converts last message to `ChatMessage`.
- SSE streaming: `/agent/stream` yields events of type `token` (LLM partials) and `message` (structured). Do not alter event keys without updating clients.

## 3. Safe Change Guidelines
- When adding a tool: implement function (pure + deterministic if possible) in an appropriate `dft_tools/*` module, wrap with LangChain `@tool`, register in `tool_registry.py`, and ensure any file writes go inside thread workspace.
- When modifying the DFT agent graph: keep the routing node names (`chat`, `dft_agent`, `tools`) unless you update references and tests/clients.
- Avoid blocking operations in async code—wrap CPU/file heavy sync calls in a thread executor if introducing new ones.
- Do not store large binary or verbose outputs directly in `ToolMessage.content`; prefer writing to a workspace file and returning a short path reference.

## 4. Common Workflows
- Add new LLM model: extend enum in `core/models.py`, map to provider string in `_MODEL_TABLE` in `agents/llm.py`, ensure API key exists (or handled if local). Update docs if externally visible.
- Add agent: create graph module (mirroring `chatbot.py` style), compile with `MemorySaver` or shared saver, register in `agent_manager.py`.
- Introduce plan/workflow logic: Reuse or extend simple `Plan`/`PlanStep` pattern in `library/dft_agent/agent.py`. Keep serializable (dataclass or pydantic) if persisting.
- Extend API: Add router under `backend/api/endpoints/`, include in `api/main.py`. Use existing `ChatMessage` / `UserInput` schemas for agent interactions.

## 5. Testing & Quality (Expected Practices)
- Prefer pure functions with small surface area for new utilities.
- Add quick tests (pytest) when modifying enums, tool registry, or workspace logic.
- Use Ruff formatting & linting (configured in project) before committing.

## 6. Performance & State Tips
- Keep agent state minimal: only essential context (workspace path, last results summary). Large artifacts -> filesystem.
- Reuse cached model instances (`@cache` in `llm.get_model`). Dont create new model objects in tight loops.
- Large prompts: avoid concatenating unbounded history; rely on LangGraph checkpoint trimming if implemented.

## 7. Security & Safety
- `python_repl` tool executes arbitrary code—restrict or disable in production pathways by gating with `settings.MODE` if adjusting behavior.
- Sanitize/validate any user-supplied filenames or path fragments—must remain inside workspace.
- Never log raw API keys; access via `SecretStr` methods (`get_secret_value()` only where required).

## 8. File & Directory References
- Core entrypoints: `backend/run_service.py`, `backend/api/main.py`.
- Workspace root: `settings.ROOT_PATH/WORKSPACE/`.
- Logs: `logs/` (ensure rotation if adding verbose logs).
- Checkpoint DB: repo root `checkpoints.db` (shared across agents).

## 9. Example: Adding a New DFT Tool (Summary)
1. Implement function in `backend/agents/dft_tools/new_tool.py` with `@tool` decorator.
2. Return small JSON-serializable dict; write bulky outputs to a workspace file.
3. Import and register name in `library/dft_agent/tool_registry.py`.
4. (Optional) Update system instructions if user-facing capability needs mention.
5. Add a smoke test ensuring the tool appears in `TOOL_REGISTRY`.

## 10. Anti-Patterns to Avoid
- Hardcoding provider model strings scattered across modules.
- Expanding state with ephemeral or large data blobs.
- Blocking synchronous heavy tasks inside async nodes without offloading.
- Returning raw filesystem paths outside the workspace.

---
If you need clarification (e.g., on plan persistence, multi-agent scaling, or tool result storage), request targeted guidance before large refactors.
