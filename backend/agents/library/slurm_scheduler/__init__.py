"""
SLURM Job Scheduler Agent

An intelligent agent for managing SLURM job submissions and monitoring in HPC environments.
Integrates with the DFT workflow to automate job scheduling and tracking.
"""

from .agent import slurm_scheduler_agent

__all__ = ["slurm_scheduler_agent"]
