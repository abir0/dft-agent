"""
Execution Utilities for DFT Calculations

Utilities for running DFT codes, validating binaries, and managing job execution.
Adapted from PR with workspace security and error handling improvements.
"""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_core.tools import tool


def _ensure_under_workspace(path: str) -> None:
    """Ensure path is under workspace root for security."""
    # Get workspace root from environment or settings
    workspace_root = os.environ.get("WORKSPACE_ROOT", "/tmp/dft_workspace")

    abs_workspace = os.path.abspath(workspace_root)
    abs_path = os.path.abspath(path)

    if not abs_path.startswith(abs_workspace):
        raise ValueError(
            f"Path must be under workspace root ({abs_workspace}); got {abs_path}"
        )


def _expand_env(cmd: str) -> str:
    """Expand environment variables in command string."""
    return os.path.expandvars(cmd)


@tool
def run_local_command(
    cmd: Union[str, List[str]],
    workdir: str,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    _thread_id: Optional[str] = None,
) -> str:
    """Run a shell command in the specified directory.

    Args:
        cmd: Command to run (string or list of arguments)
        workdir: Working directory for command execution
        env: Additional environment variables
        timeout: Command timeout in seconds
        _thread_id: Thread ID for workspace management

    Returns:
        Combined stdout/stderr output

    Raises:
        ValueError: If workdir is outside workspace root
        subprocess.CalledProcessError: If command fails
        subprocess.TimeoutExpired: If command times out
    """
    try:
        # Security check
        _ensure_under_workspace(workdir)

        # Create directory if it doesn't exist
        os.makedirs(workdir, exist_ok=True)

        # Prepare command
        if isinstance(cmd, str):
            cmd = _expand_env(cmd)
            argv = shlex.split(cmd)
        else:
            argv = cmd

        # Prepare environment
        proc_env = os.environ.copy()
        if env:
            proc_env.update({k: str(v) for k, v in env.items()})

        # Execute command
        result = subprocess.run(
            argv,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            env=proc_env,
            timeout=timeout,
        )

        return result.stdout

    except subprocess.CalledProcessError as e:
        return f"Command failed with exit code {e.returncode}:\\n{e.stdout}"
    except subprocess.TimeoutExpired as e:
        return f"Command timed out after {timeout} seconds:\\n{e.stdout or 'No output'}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def find_executable(name: str) -> str:
    """Find executable in system PATH.

    Args:
        name: Executable name (e.g., 'pw.x', 'vasp_std')

    Returns:
        Full path to executable or error message
    """
    try:
        # Use shutil.which for cross-platform compatibility
        import shutil

        path = shutil.which(name)

        if path:
            return f"Found {name}: {path}"
        else:
            # Manual search as fallback
            for p in os.environ.get("PATH", "").split(os.pathsep):
                if not p:
                    continue
                candidate = os.path.join(p, name)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return f"Found {name}: {candidate}"

            return f"Error: Executable '{name}' not found in PATH"

    except Exception as e:
        return f"Error searching for {name}: {str(e)}"


@tool
def validate_dft_binaries(
    require_qe: bool = False,
    require_vasp: bool = False,
    qe_executable: str = "pw.x",
    vasp_executable: str = "vasp_std",
) -> str:
    """Validate that required DFT binaries are available.

    Args:
        require_qe: Check for Quantum ESPRESSO binary
        require_vasp: Check for VASP binary
        qe_executable: QE executable name to check
        vasp_executable: VASP executable name to check

    Returns:
        Summary of available binaries or error messages
    """
    try:
        results = []

        if require_qe:
            # Check environment variable first
            qe_bin = os.environ.get("QE_BIN", qe_executable)
            qe_result = find_executable(qe_bin)

            if "Error:" in qe_result:
                return f"QE validation failed: {qe_result}"
            else:
                results.append(f"QE: {qe_result}")

        if require_vasp:
            # Check environment variable first
            vasp_bin = os.environ.get("VASP_BIN", vasp_executable)
            vasp_result = find_executable(vasp_bin)

            if "Error:" in vasp_result:
                return f"VASP validation failed: {vasp_result}"
            else:
                results.append(f"VASP: {vasp_result}")

        if not results:
            return "No binary validation requested"

        return "Binary validation successful:\\n" + "\\n".join(results)

    except Exception as e:
        return f"Error validating binaries: {str(e)}"


@tool
def check_calculation_status(workdir: str, calculation_type: str = "auto") -> str:
    """Check the status of a DFT calculation.

    Args:
        workdir: Directory containing calculation files
        calculation_type: Type of calculation ('qe', 'vasp', 'auto')

    Returns:
        Status summary including completion, convergence, and errors
    """
    try:
        workdir_path = Path(workdir)
        if not workdir_path.exists():
            return f"Error: Directory {workdir} does not exist"

        status_info = []

        # Auto-detect calculation type
        if calculation_type == "auto":
            if (workdir_path / "OUTCAR").exists():
                calculation_type = "vasp"
            elif any(workdir_path.glob("*.pwo")) or any(workdir_path.glob("*.out")):
                calculation_type = "qe"
            else:
                return "Cannot determine calculation type from files"

        # Check VASP calculation
        if calculation_type == "vasp":
            outcar = workdir_path / "OUTCAR"
            if outcar.exists():
                with open(outcar, "r", errors="ignore") as f:
                    content = f.read()

                if "reached required accuracy" in content:
                    status_info.append("✅ Electronic SCF converged")
                elif "EDIFF is reached" in content:
                    status_info.append("✅ Electronic SCF converged (EDIFF)")
                else:
                    status_info.append("❌ Electronic SCF not converged")

                if "TOTAL-FORCE (eV/Ang)" in content:
                    status_info.append("✅ Forces calculated")

                if "General timing and accounting informations" in content:
                    status_info.append("✅ Calculation completed normally")
                else:
                    status_info.append("❌ Calculation incomplete or crashed")
            else:
                return "VASP OUTCAR file not found"

        # Check QE calculation
        elif calculation_type == "qe":
            output_files = list(workdir_path.glob("*.pwo")) + list(
                workdir_path.glob("*.out")
            )
            if output_files:
                with open(output_files[0], "r", errors="ignore") as f:
                    content = f.read()

                if "convergence has been achieved" in content:
                    status_info.append("✅ Electronic SCF converged")
                elif "End of self-consistent calculation" in content:
                    status_info.append("✅ SCF completed")
                else:
                    status_info.append("❌ Electronic SCF not converged")

                if "JOB DONE" in content:
                    status_info.append("✅ Calculation completed normally")
                elif "%%%%%%%%%" in content:
                    status_info.append("❌ QE error encountered")
                else:
                    status_info.append("❓ Calculation status unclear")
            else:
                return "QE output file not found"

        return f"Calculation status ({calculation_type}):\\n" + "\\n".join(status_info)

    except Exception as e:
        return f"Error checking calculation status: {str(e)}"


@tool
def setup_calculation_directory(
    base_dir: str, job_name: str, _thread_id: Optional[str] = None
) -> str:
    """Set up a clean directory for DFT calculations.

    Args:
        base_dir: Base directory for calculations
        job_name: Name of the calculation job
        _thread_id: Thread ID for workspace management

    Returns:
        Path to created calculation directory
    """
    try:
        # Security check
        _ensure_under_workspace(base_dir)

        # Create calculation directory
        calc_dir = Path(base_dir) / job_name
        calc_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        subdirs = ["input", "output", "structures", "scripts"]
        for subdir in subdirs:
            (calc_dir / subdir).mkdir(exist_ok=True)

        # Create a simple README
        readme_content = f"""# DFT Calculation: {job_name}

Directory Structure:
- input/     : Input files (POSCAR, INCAR, KPOINTS, etc.)
- output/    : Output files (OUTCAR, *.pwo, etc.)
- structures/: Structure files (initial, optimized, etc.)
- scripts/   : Job submission scripts

Created: {os.path.basename(__file__)}
"""
        (calc_dir / "README.md").write_text(readme_content)

        return f"Created calculation directory: {calc_dir}"

    except Exception as e:
        return f"Error setting up directory: {str(e)}"
