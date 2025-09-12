#!/usr/bin/env python3
"""
SLURM Scheduler Agent Usage Examples

This script demonstrates how to use the SLURM Scheduler Agent for
automated HPC job management in DFT calculations.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from backend.agents.agent_manager import get_agent


async def example_basic_job_submission():
    """Example 1: Basic job submission workflow."""
    print("🚀 Example 1: Basic Job Submission")
    print("=" * 50)

    # Get the SLURM scheduler agent
    scheduler = get_agent("slurm_scheduler")

    # Create a simple job submission request
    result = await scheduler.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
        Generate and submit a SLURM job for a Quantum ESPRESSO SCF calculation.
        
        Job details:
        - Job name: pt_slab_scf
        - Input file: scf.in
        - Output file: scf.out
        - Partition: GPU-S
        - Cores: 16
        - Time limit: 2 hours
        - Memory: 32G
        """
                )
            ],
            "thread_id": "example_001",
        }
    )

    print("Agent Response:")
    print(result["messages"][-1].content)
    print()


async def example_job_monitoring():
    """Example 2: Job monitoring and status checking."""
    print("📊 Example 2: Job Monitoring")
    print("=" * 50)

    scheduler = get_agent("slurm_scheduler")

    # Monitor jobs in a workspace
    result = await scheduler.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
        Check the status of all jobs in my workspace and provide a detailed summary.
        Include job IDs, status, runtime, and any relevant information.
        """
                )
            ],
            "thread_id": "example_001",
        }
    )

    print("Monitoring Results:")
    print(result["messages"][-1].content)
    print()


async def example_convergence_workflow():
    """Example 3: Automated convergence study workflow."""
    print("🔬 Example 3: Convergence Study Workflow")
    print("=" * 50)

    scheduler = get_agent("slurm_scheduler")

    # Submit multiple jobs for convergence testing
    result = await scheduler.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
        Set up a convergence study for cutoff energy testing.
        
        Requirements:
        - Test cutoff energies: 30, 40, 50, 60 Ry
        - Each job should run for 1 hour maximum
        - Use 8 cores per job
        - Job names: cutoff_30ry, cutoff_40ry, cutoff_50ry, cutoff_60ry
        - Input files: scf_30ry.in, scf_40ry.in, scf_50ry.in, scf_60ry.in
        
        Generate all SLURM scripts and submit them to the queue.
        """
                )
            ],
            "thread_id": "convergence_study",
        }
    )

    print("Convergence Study Setup:")
    print(result["messages"][-1].content)
    print()


async def example_job_management():
    """Example 4: Job management operations."""
    print("⚙️ Example 4: Job Management")
    print("=" * 50)

    scheduler = get_agent("slurm_scheduler")

    # Demonstrate job management operations
    result = await scheduler.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
        Show me all available job management operations:
        1. List all jobs in the queue
        2. Check status of specific jobs
        3. Cancel a job if needed
        4. Retrieve job outputs
        5. Monitor job progress
        
        Provide examples of each operation.
        """
                )
            ],
            "thread_id": "job_management_demo",
        }
    )

    print("Job Management Operations:")
    print(result["messages"][-1].content)
    print()


async def example_integration_with_dft_agent():
    """Example 5: Integration with DFT Agent."""
    print("🔗 Example 5: Integration with DFT Agent")
    print("=" * 50)

    # Get both agents
    dft_agent = get_agent("dft_agent")
    scheduler = get_agent("slurm_scheduler")

    # First, use DFT agent to generate a structure and QE input
    print("Step 1: Generate structure and QE input with DFT Agent...")
    dft_result = await dft_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
        Generate a Pt(111) slab structure and create a QE SCF input file.
        Use 50 Ry cutoff energy and appropriate k-points.
        Save the input file as 'pt_slab_scf.in'.
        """
                )
            ],
            "thread_id": "integrated_workflow",
        }
    )

    print("DFT Agent Response:")
    print(dft_result["messages"][-1].content[:200] + "...")
    print()

    # Then, use SLURM scheduler to submit the job
    print("Step 2: Submit job with SLURM Scheduler...")
    scheduler_result = await scheduler.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
        Now submit the QE calculation to SLURM.
        Use the input file 'pt_slab_scf.in' that was just created.
        Allocate 16 cores for 2 hours on the GPU-S partition.
        """
                )
            ],
            "thread_id": "integrated_workflow",
        }
    )

    print("SLURM Scheduler Response:")
    print(scheduler_result["messages"][-1].content)
    print()


async def main():
    """Run all examples."""
    print("🧪 SLURM Scheduler Agent Examples")
    print("=" * 60)
    print()

    try:
        await example_basic_job_submission()
        await example_job_monitoring()
        await example_convergence_workflow()
        await example_job_management()
        await example_integration_with_dft_agent()

        print("✅ All examples completed successfully!")
        print()
        print("💡 Tips:")
        print("- Use the web interface at http://localhost:8501 for interactive usage")
        print("- Check the API documentation at http://localhost:8083/docs")
        print("- Review the full documentation in docs/SLURM_SCHEDULER.md")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure the DFT Agent services are running:")
        print("  - Backend: http://localhost:8083")
        print("  - Frontend: http://localhost:8501")


if __name__ == "__main__":
    asyncio.run(main())
