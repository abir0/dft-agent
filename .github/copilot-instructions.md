# Copilot Project Instructions

Concise, actionable guidance for AI coding agents working in this repo. Focus on preserving existing architecture, following established patterns, and avoiding unsafe changes.

## 1. Architecture Snapshot
- Backend service: FastAPI (`backend/api/main.py`) exposing agent endpoints under `/agent/*` plus `/health` and `/info`.
- Agents built with LangGraph (`backend/agents/`):
  - `library/chatbot.py`: General-purpose conversational agent (web search, calculator, python REPL, literature tools, optional MCP scholarly tools).
  - `library/dft_agent/agent.py`: Fully implemented DFT workflow agent with routing (chat vs dft_agent vs tools), planning, state management.
  - Tool registry: `library/dft_agent/tool_registry.py` + DFT tools in `backend/agents/dft_tools/*.py` (structure, QE inputs, convergence, Materials Project, database, execution).
- Agent registry: `backend/agents/agent_manager.py` maps agent keys to compiled graphs.
- LLM provider abstraction: `backend/agents/llm.py` + model enums in `backend/core/models.py`; dynamic selection via `settings.DEFAULT_MODEL`.
- State checkpointing: SQLite (`checkpoints.db`) via `AsyncSqliteSaver` configured in `api/main.py` lifespan.
- Workspace & per-thread file organization: `backend/utils/workspace.py`; thread/session ID embedded in LangGraph config (`configurable.thread_id`).
- Schemas & API data contracts: `backend/core/schema.py`.
 - Default agent is `dft_agent` (see `DEFAULT_AGENT` constant).
 - Additional endpoints: `/agent/feedback` (LangSmith), `/agent/history` (thread replay), `/agent/stream` (SSE tokens + structured messages).

## 2. Key Conventions & Patterns
- Thread/Conversation continuity: Always pass `thread_id` (or let API create one). Workspace dirs live under `WORKSPACE/<thread_id>/` with standardized subfolders.
- Tools: Added by binding via `model.bind_tools([...])`; DFT agent manually processes tool calls in `custom_tool_node` to inject `thread_id`.
- Prompts: Large system instructions embedded inline; if refactoring, extract to a module but keep semantic content stable.
- Models: Use enums (`OpenAIModelName`, `GroqModelName`, etc.)—never hardcode raw model strings outside `llm.py` or enums.
- Settings: Centralized in `backend/settings.py`; do not mutate global `settings` except via environment. Use `settings.is_dev()` for dev-only behavior.
- Return shape for invoke endpoints: `{"messages": [...]}` from graph; API converts last message to `ChatMessage`.
- SSE streaming: `/agent/stream` yields events of type `token` (LLM partials) and `message` (structured). Do not alter event keys without updating clients.
 - Planning: Simple `Plan` objects live in DFT agent state; plan operations triggered by natural language ("create plan", "execute plan"). Keep steps deterministic & tool-referential.
 - Tool outputs: Return concise summaries; large artifacts must be stored in workspace and referenced by relative path.

## 3. Safe Change Guidelines
- When adding a tool: implement function (pure + deterministic if possible) in an appropriate `dft_tools/*` module, wrap with LangChain `@tool`, register in `tool_registry.py`, and ensure any file writes go inside thread workspace.
- When modifying the DFT agent graph: keep the routing node names (`chat`, `dft_agent`, `tools`) unless you update references and tests/clients.
- Avoid blocking operations in async code—wrap CPU/file heavy sync calls in a thread executor if introducing new ones.
- Do not store large binary or verbose outputs directly in `ToolMessage.content`; prefer writing to a workspace file and returning a short path reference.
 - Preserve SSE contract: events only `token`, `message`, and final `[DONE]` sentinel (client assumptions rely on this).
 - When extending planning, maintain backward compatibility for existing plan commands; add new commands via natural language interpretation rather than rigid keywords where possible.
 - Do not expand agent state with large raw data; always offload to files.

## 4. Common Workflows
- Add new LLM model: extend enum in `core/models.py`, map to provider string in `_MODEL_TABLE` in `agents/llm.py`, ensure API key exists (or handled if local). Update docs if externally visible.
- Add agent: create graph module (mirroring `chatbot.py` style), compile with `MemorySaver` or shared saver, register in `agent_manager.py`.
- Introduce plan/workflow logic: Reuse or extend simple `Plan`/`PlanStep` pattern in `library/dft_agent/agent.py`. Keep serializable (dataclass or pydantic) if persisting.
- Extend API: Add router under `backend/api/endpoints/`, include in `api/main.py`. Use existing `ChatMessage` / `UserInput` schemas for agent interactions.

## 5. Testing & Quality (Expected Practices)
- Prefer pure functions with small surface area for new utilities.
- Add quick tests (pytest) when modifying enums, tool registry, or workspace logic.
- Use Ruff formatting & linting (configured in project) before committing.
 - For new tools: add a smoke test asserting the tool name appears in `TOOL_REGISTRY` and basic call returns a non-error string/dict.
 - Use the `Fake` model enum for deterministic tests when LLM output isn't the focus.
 - Keep test runtime short (<5s typical) by stubbing external calls (e.g., avoid real network or heavy DFT runs). If unavoidable, mark with a skip/slow marker.

## 6. Performance & State Tips
- Keep agent state minimal: only essential context (workspace path, last results summary). Large artifacts -> filesystem.
 - Reuse cached model instances (`@cache` in `llm.get_model`). Do not create new model objects in tight loops.
- Large prompts: avoid concatenating unbounded history; rely on LangGraph checkpoint trimming if implemented.
 - For CPU-intensive structure transformations, consider batching or async offloading if added in future.
 - Prefer incremental convergence test strategies (reuse prior results) if implementing more advanced logic.

## 7. Security & Safety
- `python_repl` tool executes arbitrary code—restrict or disable in production pathways by gating with `settings.MODE` if adjusting behavior.
- Sanitize/validate any user-supplied filenames or path fragments—must remain inside workspace.
- Never log raw API keys; access via `SecretStr` methods (`get_secret_value()` only where required).

## 8. File & Directory References
- Core entrypoints: `backend/run_service.py`, `backend/api/main.py`.
- Workspace root: `settings.ROOT_PATH/WORKSPACE/`.
- Logs: `logs/` (ensure rotation if adding verbose logs).
- Checkpoint DB: repo root `checkpoints.db` (shared across agents).
 - DFT tools root: `backend/agents/dft_tools/` (group by domain if adding more: e.g. `vasp_tools.py`).
 - Planning code: `backend/agents/library/dft_agent/agent.py` (simple Plan classes inline).

## 9. Example: Adding a New DFT Tool (Summary)
1. Implement function in `backend/agents/dft_tools/new_tool.py` with `@tool` decorator.
2. Return small JSON-serializable dict; write bulky outputs to a workspace file.
3. Import and register name in `library/dft_agent/tool_registry.py`.
4. (Optional) Update system instructions if user-facing capability needs mention.
5. Add a smoke test ensuring the tool appears in `TOOL_REGISTRY`.
 6. Update README / CLAUDE.md if it introduces a new capability class (e.g., a new engine interface).

## 10. Anti-Patterns to Avoid
- Hardcoding provider model strings scattered across modules.
- Expanding state with ephemeral or large data blobs.
- Blocking synchronous heavy tasks inside async nodes without offloading.
- Returning raw filesystem paths outside the workspace.
 - Embedding entire file contents (logs, large JSON) in tool message content instead of referencing a saved file.
 - Introducing hidden side effects (network, filesystem) in tools without `_thread_id` scoping.

## 11. Planning System Notes
- Current planning is linear & sequential; each step maps to a tool name + arguments.
- Extend by adding lightweight validation helpers or a planner module (avoid heavy reasoning loops in the runtime path).
- Steps should be idempotent when possible; if not, document side effects (e.g., file overwrite) in tool docstring.
- If adding plan serialization, prefer a small pydantic model or dataclass and store in workspace JSON.

## 12. Adding New Endpoints
- Place new routers under `backend/api/endpoints/` and include them in `api/main.py`.
- Reuse existing schemas in `backend/core/schema.py` where possible.
- Maintain consistent response envelope style; avoid introducing divergent structures unless justified.

## 13. Model Extension Checklist
1. Add enum value in `backend/core/models.py`.
2. Map to provider string in `_MODEL_TABLE` in `backend/agents/llm.py`.
3. Ensure environment variable / key path is documented.
4. Update README model support section.
5. (Optional) Add a simple invoke test using the fake model fallback if real key not present.

---
If you need clarification (e.g., on plan persistence, multi-agent scaling, or tool result storage), request targeted guidance before large refactors.
