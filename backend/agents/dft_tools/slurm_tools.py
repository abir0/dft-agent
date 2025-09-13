"""
SLURM Job Scheduler Tools

Tools for submitting, monitoring, and managing SLURM jobs in HPC environments.
Integrates with the DFT workflow to automate job submission and tracking.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool

from backend.utils.workspace import get_subdir_path


class SLURMJobTemplate:
    """Template for SLURM job scripts with customizable parameters."""

    DEFAULT_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --mem={memory}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Working Directory=$SLURM_SUBMIT_DIR"
echo "Job started at: $(date)"

# Load required modules
{module_commands}

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

# Change to the workspace directory where input files are located
cd {workspace_dir}

# Create output directory if it doesn't exist
mkdir -p {output_dir}

# Run the calculation
{execution_command}

echo "Job completed at: $(date)"
"""

    @classmethod
    def generate_script(
        cls,
        job_name: str,
        input_file: str,
        output_file: str,
        error_file: Optional[str] = None,
        partition: str = "GPU-S",
        ntasks_per_node: int = 16,
        cpus_per_task: int = 1,
        time_limit: str = "72:00:00",
        memory: str = "32G",
        qe_path: str = "/cm/shared/apps/quantumespresso/qe-7.2/bin",
        module_commands: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate a SLURM job script.

        Args:
            job_name: Name of the job
            input_file: QE input file name
            output_file: Output file name
            partition: SLURM partition to use
            ntasks_per_node: Number of tasks per node
            cpus_per_task: Number of CPUs per task
            time_limit: Time limit in HH:MM:SS format
            memory: Memory allocation
            qe_path: Path to Quantum ESPRESSO binaries
            module_commands: List of module load commands

        Returns:
            Complete SLURM job script as string
        """
        if module_commands is None:
            module_commands = [
                "module load compiler-rt/latest",
                "module load mkl/latest",
                "module load mpi/latest",
            ]

        # Use workspace directory if provided, otherwise use current directory
        if workspace_dir is None:
            workspace_dir = "$SLURM_SUBMIT_DIR"

        # Set output directory - use the directory containing the output file
        if output_dir is None:
            output_dir = str(Path(output_file).parent)

        # Use provided error_file or generate from output_file
        if error_file is None:
            error_file = output_file.replace(".out", ".err")
        execution_command = (
            f"mpiexec -np $SLURM_NTASKS_PER_NODE {qe_path}/pw.x -i {input_file}"
        )

        return cls.DEFAULT_TEMPLATE.format(
            partition=partition,
            job_name=job_name,
            output_file=output_file,
            error_file=error_file,
            ntasks_per_node=ntasks_per_node,
            cpus_per_task=cpus_per_task,
            time_limit=time_limit,
            memory=memory,
            module_commands="\n".join(module_commands),
            execution_command=execution_command,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
        )


@tool
def generate_slurm_script(
    job_name: str,
    input_file: str,
    output_file: str,
    partition: str = "GPU-S",
    ntasks_per_node: int = 16,
    cpus_per_task: int = 1,
    time_limit: str = "72:00:00",
    memory: str = "32G",
    qe_path: str = "/cm/shared/apps/quantumespresso/qe-7.2/bin",
    thread_id: Optional[str] = None,
) -> str:
    """Generate a SLURM job script for Quantum ESPRESSO calculations.

    Args:
        job_name: Name of the job (will be sanitized for SLURM)
        input_file: QE input file name
        output_file: Output file name
        partition: SLURM partition to use (default: GPU-S)
        ntasks_per_node: Number of tasks per node (default: 16)
        cpus_per_task: Number of CPUs per task (default: 1)
        time_limit: Time limit in HH:MM:SS format (default: 72:00:00)
        memory: Memory allocation (default: 32G)
        qe_path: Path to Quantum ESPRESSO binaries
        thread_id: Thread ID for workspace organization

    Returns:
        Path to the generated SLURM script file
    """
    try:
        # Sanitize job name for SLURM
        safe_job_name = "".join(c for c in job_name if c.isalnum() or c in "_-")[:50]

        # Get workspace directory
        if thread_id:
            workspace_dir = get_subdir_path(thread_id, "calculations/jobs")
        else:
            workspace_dir = Path.cwd() / "calculations" / "jobs"
            workspace_dir.mkdir(parents=True, exist_ok=True)

        # Get workspace directory for the thread
        workspace_dir_path = None
        if thread_id:
            from backend.utils.workspace import get_workspace_path

            workspace_path = get_workspace_path(thread_id)
            workspace_dir_path = str(workspace_path)

        # For SLURM output files, use simple filenames relative to workspace directory
        # Since the script changes to workspace_dir, these will be created there
        simple_output_file = f"{safe_job_name}.out"
        simple_error_file = f"{safe_job_name}.err"

        # Generate script content with absolute paths for output files
        if workspace_dir_path:
            abs_output_file = f"{workspace_dir_path}/{simple_output_file}"
            abs_error_file = f"{workspace_dir_path}/{simple_error_file}"
        else:
            abs_output_file = simple_output_file
            abs_error_file = simple_error_file

        script_content = SLURMJobTemplate.generate_script(
            job_name=safe_job_name,
            input_file=input_file,
            output_file=abs_output_file,
            error_file=abs_error_file,
            partition=partition,
            ntasks_per_node=ntasks_per_node,
            cpus_per_task=cpus_per_task,
            time_limit=time_limit,
            memory=memory,
            qe_path=qe_path,
            workspace_dir=workspace_dir_path,
            output_dir=".",
        )

        # Write script to file
        script_filename = f"{safe_job_name}.sh"
        script_path = workspace_dir / script_filename

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        return f"✅ Generated SLURM script: {script_path}\nJob name: {safe_job_name}\nInput: {input_file}\nOutput: {simple_output_file}\nError: {simple_error_file}\n\n⚠️  IMPORTANT: You must now call submit_slurm_job() to actually submit this job to the queue!"

    except Exception as e:
        return f"Error generating SLURM script: {str(e)}"


@tool
def submit_slurm_job(
    script_path: str,
    thread_id: Optional[str] = None,
) -> str:
    """Submit a SLURM job script to the queue.

    Args:
        script_path: Path to the SLURM script file
        thread_id: Thread ID for tracking

    Returns:
        Job submission result with job ID
    """
    try:
        script_path = Path(script_path)

        if not script_path.exists():
            return f"Error: Script file not found: {script_path}"

        # Submit the job
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            # Parse job ID from output
            output_lines = result.stdout.strip().split("\n")
            job_id = None
            for line in output_lines:
                if "Submitted batch job" in line:
                    job_id = line.split()[-1]
                    break

            if not job_id:
                return f"Error: Could not parse job ID from sbatch output: {result.stdout.strip()}"

            # Store job information
            if thread_id:
                job_info = {
                    "job_id": job_id,
                    "script_path": str(script_path),
                    "submitted_at": datetime.now().isoformat(),
                    "status": "PENDING",
                    "thread_id": thread_id,
                }

                # Save job info to workspace
                workspace_dir = get_subdir_path(thread_id, "calculations/jobs")
                job_info_file = workspace_dir / f"job_{job_id}.json"

                with open(job_info_file, "w") as f:
                    json.dump(job_info, f, indent=2)

            return f"✅ Job submitted successfully!\nJob ID: {job_id}\nScript: {script_path}\nOutput: {result.stdout.strip()}"
        else:
            return f"❌ Error submitting job: {result.stderr.strip()}\nReturn code: {result.returncode}"

    except subprocess.TimeoutExpired:
        return "Error: Job submission timed out"
    except Exception as e:
        return f"Error submitting job: {str(e)}"


@tool
def check_slurm_job_status(
    job_id: str,
    thread_id: Optional[str] = None,
) -> str:
    """Check the status of a SLURM job.

    Args:
        job_id: SLURM job ID
        thread_id: Thread ID for context

    Returns:
        Job status information
    """
    try:
        # Get job status using squeue
        result = subprocess.run(
            ["squeue", "-j", job_id, "--format=%i,%T,%M,%N", "--noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            if result.stdout.strip():
                # Job is still in queue
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    status = parts[1]
                    runtime = parts[2] if len(parts) > 2 else "Unknown"
                    nodes = parts[3] if len(parts) > 3 else "Unknown"

                    status_info = f"Job {job_id} Status: {status}\nRuntime: {runtime}\nNodes: {nodes}"
                else:
                    status_info = f"Job {job_id} is in queue: {result.stdout.strip()}"
            else:
                # Job not in queue - check if completed
                status_info = f"Job {job_id} not found in queue. Checking if completed..."

                # Try to get job info from sacct
                sacct_result = subprocess.run(
                    [
                        "sacct",
                        "-j",
                        job_id,
                        "--format=JobID,State,ExitCode,Start,End",
                        "--noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if sacct_result.returncode == 0 and sacct_result.stdout.strip():
                    status_info += f"\nJob history: {sacct_result.stdout.strip()}"
                else:
                    status_info += "\nNo job history found."
        else:
            status_info = f"Error checking job status: {result.stderr.strip()}"

        # Update job info file if thread_id provided
        if thread_id:
            workspace_dir = get_subdir_path(thread_id, "calculations/jobs")
            job_info_file = workspace_dir / f"job_{job_id}.json"

            if job_info_file.exists():
                try:
                    with open(job_info_file, "r") as f:
                        job_info = json.load(f)

                    # Update status
                    if "Status: " in status_info:
                        new_status = status_info.split("Status: ")[1].split("\n")[0]
                        job_info["status"] = new_status
                        job_info["last_checked"] = datetime.now().isoformat()

                        with open(job_info_file, "w") as f:
                            json.dump(job_info, f, indent=2)
                except Exception as e:
                    status_info += f"\nWarning: Could not update job info: {e}"

        return status_info

    except subprocess.TimeoutExpired:
        return "Error: Status check timed out"
    except Exception as e:
        return f"Error checking job status: {str(e)}"


@tool
def cancel_slurm_job(
    job_id: str,
    thread_id: Optional[str] = None,
) -> str:
    """Cancel a SLURM job.

    Args:
        job_id: SLURM job ID to cancel
        thread_id: Thread ID for context

    Returns:
        Cancellation result
    """
    try:
        result = subprocess.run(
            ["scancel", job_id], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            # Update job info file
            if thread_id:
                workspace_dir = get_subdir_path(thread_id, "calculations/jobs")
                job_info_file = workspace_dir / f"job_{job_id}.json"

                if job_info_file.exists():
                    try:
                        with open(job_info_file, "r") as f:
                            job_info = json.load(f)

                        job_info["status"] = "CANCELLED"
                        job_info["cancelled_at"] = datetime.now().isoformat()

                        with open(job_info_file, "w") as f:
                            json.dump(job_info, f, indent=2)
                    except Exception:
                        pass  # Don't fail the cancellation for this

            return f"Job {job_id} cancelled successfully"
        else:
            return f"Error cancelling job: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Error: Cancellation timed out"
    except Exception as e:
        return f"Error cancelling job: {str(e)}"


@tool
def list_slurm_jobs(
    user: Optional[str] = None,
    partition: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> str:
    """List SLURM jobs in the queue.

    Args:
        user: Filter by username (default: current user)
        partition: Filter by partition
        thread_id: Thread ID for context

    Returns:
        List of jobs in the queue
    """
    try:
        cmd = ["squeue", "--format=%i,%j,%T,%M,%N,%P", "--noheader"]

        if user:
            cmd.extend(["-u", user])
        if partition:
            cmd.extend(["-p", partition])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                job_list = "SLURM Jobs in Queue:\n"
                job_list += "ID\tName\tStatus\tTime\tNodes\tPartition\n"
                job_list += "-" * 60 + "\n"

                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 6:
                        job_list += f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\t{parts[4]}\t{parts[5]}\n"

                return job_list
            else:
                return "No jobs found in queue"
        else:
            return f"Error listing jobs: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Error: Job listing timed out"
    except Exception as e:
        return f"Error listing jobs: {str(e)}"


@tool
def get_slurm_job_output(
    job_id: str,
    output_file: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> str:
    """Get the output from a completed SLURM job.

    Args:
        job_id: SLURM job ID
        output_file: Specific output file to read (optional)
        thread_id: Thread ID for workspace context

    Returns:
        Job output content or file information
    """
    try:
        # If no specific output file, try to find it
        if not output_file:
            if thread_id:
                workspace_dir = get_subdir_path(thread_id, "calculations/jobs")
                job_info_file = workspace_dir / f"job_{job_id}.json"

                if job_info_file.exists():
                    with open(job_info_file, "r") as f:
                        job_info = json.load(f)

                    # Try to find output files
                    script_path = Path(job_info.get("script_path", ""))
                    if script_path.exists():
                        # Look for output files in the same directory
                        output_files = list(script_path.parent.glob(f"slurm-{job_id}.*"))
                        if output_files:
                            output_file = str(output_files[0])

        if output_file and Path(output_file).exists():
            with open(output_file, "r") as f:
                content = f.read()

            # Limit output size to prevent overwhelming response
            if len(content) > 10000:
                content = content[:10000] + "\n... (output truncated)"

            return f"Job {job_id} output:\n{content}"
        else:
            return f"No output file found for job {job_id}. Check the job directory for slurm-{job_id}.* files."

    except Exception as e:
        return f"Error getting job output: {str(e)}"


@tool
def monitor_slurm_jobs(
    thread_id: str,
    check_interval: int = 60,
    max_checks: int = 10,
) -> str:
    """Monitor SLURM jobs for a specific thread/workspace.

    Args:
        thread_id: Thread ID to monitor jobs for
        check_interval: Seconds between checks (default: 60)
        max_checks: Maximum number of checks (default: 10)

    Returns:
        Monitoring results and job status updates
    """
    try:
        workspace_dir = get_subdir_path(thread_id, "calculations/jobs")

        if not workspace_dir.exists():
            return f"No workspace found for thread {thread_id}"

        # Find all job info files
        job_files = list(workspace_dir.glob("job_*.json"))

        if not job_files:
            return f"No jobs found for thread {thread_id}"

        results = f"Monitoring {len(job_files)} jobs for thread {thread_id}:\n"

        for job_file in job_files:
            try:
                with open(job_file, "r") as f:
                    job_info = json.load(f)

                job_id = job_info.get("job_id")
                if job_id:
                    status = check_slurm_job_status(job_id, thread_id)
                    results += f"\nJob {job_id}: {status}\n"

            except Exception as e:
                results += f"\nError reading job file {job_file}: {e}\n"

        return results

    except Exception as e:
        return f"Error monitoring jobs: {str(e)}"


@tool
def verify_job_submission(thread_id: str) -> str:
    """Verify if any jobs were actually submitted for a given thread.

    Args:
        thread_id: Thread ID to check for submitted jobs

    Returns:
        Information about submitted jobs or confirmation that none were submitted
    """
    try:
        workspace_dir = get_subdir_path(thread_id, "calculations/jobs")

        if not workspace_dir.exists():
            return (
                f"No jobs directory found for thread {thread_id}. No jobs were submitted."
            )

        # Find all job info files
        job_files = list(workspace_dir.glob("job_*.json"))

        if not job_files:
            return f"No job files found in {workspace_dir}. No jobs were submitted."

        results = f"Found {len(job_files)} submitted jobs for thread {thread_id}:\n"

        for job_file in job_files:
            try:
                import json

                with open(job_file, "r") as f:
                    job_info = json.load(f)

                job_id = job_info.get("job_id", "Unknown")
                status = job_info.get("status", "Unknown")
                submitted_at = job_info.get("submitted_at", "Unknown")

                results += (
                    f"Job ID: {job_id}, Status: {status}, Submitted: {submitted_at}\n"
                )

            except Exception as e:
                results += f"• Error reading job file {job_file}: {e}\n"

        return results

    except Exception as e:
        return f"Error verifying job submission: {str(e)}"
