from fastapi import APIRouter
from pydantic import BaseModel
from backend.agents import get_agent

router = APIRouter(prefix="/dft-planner", tags=["dft"])

class PlanRequest(BaseModel):
    request: str
    hints: dict | None = None
    code: dict | None = None

@router.post("/plan")
async def plan_endpoint(body: PlanRequest):
    agent = get_agent("dft_planner")
    out = agent.plan(body.request, hints=body.hints, code=body.code)
    return {"request": body.model_dump(), **out}
