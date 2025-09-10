
from .tool_registry import TOOL_REGISTRY
import json

def execute_node(state: dict) -> dict:
    plan = state.get("plan")
    if not plan:
        raise ValueError("No plan found in the state to execute.")

    step_outputs = {}
    for i, step in enumerate(plan.get("steps", [])):
        step_num = i + 1
        tool_name = step.get("tool")
        args = step.get("args", {})
        if tool_name not in TOOL_REGISTRY:
            return {"error": f"Tool '{tool_name}' not found."}

        tool_function = TOOL_REGISTRY[tool_name]

        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$outputs["):
                ref_step = int(value[9:-1])
                args[key] = step_outputs[ref_step]

        try:
            result = tool_function(**args)
            step_outputs[step_num] = result
        except Exception as e:
            return {"error": f"Error in step {step_num} ('{tool_name}'): {e}"}

        return {"execution_results": step_outputs}




