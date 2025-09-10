import re, json, copy

def normalize_kpts(v):
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "x").replace("*", "x")
        if re.fullmatch(r"\d+x\d+x\d+", s):
            return s
        if s.isdigit():
            return f"{s}x{s}x{s}"
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        a, b, c = [int(x) for x in v]
        return f"{a}x{b}x{c}"
    return v

def validate_plan_schema(plan: dict) -> list[str]:
    errs = []
    if not isinstance(plan, dict):
        return ["Plan must be a JSON object"]
    if "goal" not in plan:
        errs.append("Missing key: goal")
    if "steps" not in plan:
        errs.append("Missing key: steps")
        return errs

    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        errs.append("steps must be a non-empty list")
        return errs

    code = (plan.get("inputs_summary") or {}).get("calculation_code") \
        or (plan.get("inputs_summary") or {}).get("code") \
        or (plan.get("code") or "")
    code = str(code).lower()

    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            errs.append(f"Step {i}: must be an object")
            continue
        tool = s.get("tool")
        args = s.get("args")
        if not isinstance(tool, str) or not tool:
            errs.append(f"Step {i}: missing 'tool' (string)")
        if not isinstance(args, dict):
            errs.append(f"Step {i}: missing 'args' (object)")
            continue

        if "kpts" in args:
            args["kpts"] = normalize_kpts(args["kpts"])

        if code == "vasp" and tool and tool.startswith("write_qe_"):
            errs.append(f"Step {i}: uses QE tool '{tool}' but code=VASP")
        if code == "qe" and tool and tool.startswith("write_vasp_"):
            errs.append(f"Step {i}: uses VASP tool '{tool}' but code=QE")

        if tool in {"write_vasp_neb", "neb_setup"}:
            imgs = args.get("images")
            if not imgs or not isinstance(imgs, list) or len(imgs) < 2:
                errs.append(f"Step {i}: NEB requires ≥2 images (initial & final)")

        if tool == "move_atom":
            if "species" not in args:
                errs.append(f"Step {i}: move_atom needs 'species'")
            if "from_index" not in args and "from_fractional" not in args:
                errs.append(f"Step {i}: move_atom needs 'from_index' or 'from_fractional'")
            if "to_neighbor_index" not in args and "to_fractional" not in args:
                errs.append(f"Step {i}: move_atom needs 'to_neighbor_index' or 'to_fractional'")

    return errs

def diff_plans(old: dict, new: dict) -> list[str]:
    changes = []
    if (old or {}).get("goal") != (new or {}).get("goal"):
        changes.append("goal changed")
    os = (old or {}).get("steps", [])
    ns = (new or {}).get("steps", [])
    if len(os) != len(ns):
        changes.append(f"step count: {len(os)} → {len(ns)}")
    for i, (a, b) in enumerate(zip(os, ns), 1):
        if a.get("tool") != b.get("tool"):
            changes.append(f"step {i} tool: {a.get('tool')} → {b.get('tool')}")
        if a.get("args") != b.get("args"):
            changes.append(f"step {i} args changed")
    return changes