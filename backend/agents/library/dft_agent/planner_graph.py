from __future__ import annotations
import json, os
from pathlib import Path
from backend.agents.llm import get_model
from backend.settings import settings
from .tool_inventory import tools_text_block
from backend.core import AllModelEnum
PROMPT_PATH = Path(__file__).with_name("system_prompt_planner.md")
model_name = settings.DEFAULT_MODEL
try:
    model_name = AllModelEnum(model_name)
except Exception:
    pass

def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def _assemble_prompt(workroot: str) -> str:
    core = _load_prompt()
    core = core.replace("{TOOLS}", tools_text_block())
    core = core.replace("{WR}", workroot)
    return core

def _json_only(text: str) -> dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i:j+1])
    raise ValueError("Planner did not return JSON.")

def generate_plan(request_text: str, hints: dict | None = None, code: dict | None = None, history: list | None = None) -> dict:
    wr = os.environ.get("WORKSPACE_ROOT", "/app/workspace")
    workroot = os.path.join(wr, f"plan_{abs(hash(request_text)) % 10**10}")
    system_prompt = _assemble_prompt(workroot)
    formatted_history = []
    if history:
        for msg in history:
            role = "user" if msg.get("type") == "human" else "assistant"
            formatted_history.append({"role": role, "content": msg.get("content", "")})
    user_payload = {"request": request_text, "hints": hints or {}, "code": code or {}}
    final_user_message = {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    chat = get_model(model_name)
    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        *formatted_history,  # Unpack the formatted history here
        final_user_message
    ]
    for i in range(3):
        resp = chat.invoke(messages_for_llm)

        plan = _json_only(resp.content)

        # minimal schema guardrails
        if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Invalid plan JSON: missing or malformed 'steps' array.")
        if "goal" not in plan:
            raise ValueError("Invalid plan JSON: missing 'goal'.")

    return {"plan": plan, "workroot": workroot}
