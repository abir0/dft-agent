"""
SLURM Scheduler Agent Implementation

An intelligent agent that manages SLURM job submissions, monitoring, and workflow automation
for DFT calculations in HPC environments.
"""

from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.agents.dft_tools.slurm_tools import (
    generate_slurm_script,
    submit_slurm_job,
    check_slurm_job_status,
    cancel_slurm_job,
    list_slurm_jobs,
    get_slurm_job_output,
    monitor_slurm_jobs,
)
from backend.agents.llm import get_model
from backend.settings import settings


class SLURMSchedulerState(TypedDict):
    """State for the SLURM Scheduler Agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: Optional[str]
    current_jobs: List[Dict]
    workflow_status: str


def create_slurm_scheduler_agent() -> StateGraph:
    """Create the SLURM Scheduler Agent graph."""
    
    # Define the tools available to the scheduler
    tools = [
        generate_slurm_script,
        submit_slurm_job,
        check_slurm_job_status,
        cancel_slurm_job,
        list_slurm_jobs,
        get_slurm_job_output,
        monitor_slurm_jobs,
    ]
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # System prompt for the SLURM scheduler
    system_prompt = """You are a SLURM Job Scheduler Agent specialized in managing HPC job submissions and monitoring for DFT calculations.

Your capabilities include:
1. **Job Script Generation**: Create SLURM job scripts with appropriate resource allocation
2. **Job Submission**: Submit jobs to the SLURM queue with proper error handling
3. **Job Monitoring**: Track job status, runtime, and completion
4. **Job Management**: Cancel jobs, list queue status, and retrieve outputs
5. **Workflow Integration**: Coordinate with DFT calculations and other agents

Key responsibilities:
- Generate SLURM scripts with optimal resource allocation for QE calculations
- Submit jobs safely with proper error checking
- Monitor job progress and provide status updates
- Handle job failures and provide troubleshooting guidance
- Maintain job history and workspace organization
- Integrate seamlessly with DFT workflow agents

Best practices:
- Always validate job parameters before submission
- Use appropriate resource allocation (CPUs, memory, time limits)
- Provide clear status updates and error messages
- Maintain job tracking information in workspace
- Handle SLURM-specific errors gracefully
- Coordinate with other agents for complete workflows

When users ask about job scheduling, always:
1. Understand the calculation requirements
2. Generate appropriate SLURM scripts
3. Submit jobs with proper tracking
4. Monitor progress and provide updates
5. Handle any issues that arise

Be proactive in monitoring jobs and providing status updates. Always maintain clear communication about job status and any issues that need attention."""

    def scheduler_node(state: SLURMSchedulerState) -> SLURMSchedulerState:
        """Main scheduler node that processes user requests."""
        
        # Get the latest user message
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if not last_message or not isinstance(last_message, HumanMessage):
            return state
        
        # Get LLM for reasoning
        llm = get_model(settings.DEFAULT_MODEL)
        
        # Create system message
        system_msg = SystemMessage(content=system_prompt)
        
        # Prepare messages for LLM
        llm_messages = [system_msg] + messages
        
        # Get LLM response
        response = llm.invoke(llm_messages)
        
        # Add response to state
        state["messages"].append(response)
        
        return state

    def should_continue(state: SLURMSchedulerState) -> str:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return END

    # Create the graph
    workflow = StateGraph(SLURMSchedulerState)
    
    # Add nodes
    workflow.add_node("scheduler", scheduler_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge("tools", "scheduler")
    workflow.add_conditional_edges(
        "scheduler",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    
    # Set entry point
    workflow.set_entry_point("scheduler")
    
    # Compile the graph
    return workflow.compile()


# Create the agent instance
slurm_scheduler_agent = create_slurm_scheduler_agent()
