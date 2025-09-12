from __future__ import annotations
import os
import shlex
import subprocess
from typing import Dict, List, Union

from langchain_core.tools import tool


def _ensure_under_workspace(path: str) -> None:
    """Raise if path escapes WORKSPACE_ROOT."""
    wr = os.environ.get("WORKSPACE_ROOT", "/app/workspace")
    abs_wr = os.path.abspath(wr)
    abs_p = os.path.abspath(path)
    if not abs_p.startswith(abs_wr):
        raise ValueError(f"workdir must be under WORKSPACE_ROOT ({abs_wr}); got {abs_p}")


def _expand_env(cmd: str) -> str:
    """Expand $VARS in a command string using current environment."""
    return os.path.expandvars(cmd)


@tool
def run_local(cmd: Union[str, List[str]], workdir: str, env: Dict[str, str] | None = None) -> str:
    """
    Run a shell command in the given working directory and return combined stdout/stderr.

    Args:
        cmd: Command to run. If a string, it's split with shlex (respecting quotes).
             Environment variables like $QE_BIN are expanded automatically.
        workdir: Directory where the command is executed. Must live under WORKSPACE_ROOT.
        env: Optional extra environment variables to inject/override for this command.

    Returns:
        The process output (stdout with stderr merged). Raises CalledProcessError on non-zero exit.
    """
    _ensure_under_workspace(workdir)
    os.makedirs(workdir, exist_ok=True)

    # Prepare command
    if isinstance(cmd, str):
        cmd = _expand_env(cmd)
        argv = shlex.split(cmd)
    else:
        argv = cmd

    # Environment
    proc_env = os.environ.copy()
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})

    # Execute
    proc = subprocess.run(
        argv,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        env=proc_env,
    )
    return proc.stdout


@tool
def which_bin(name: str) -> str:
    """
    Return the full path for an executable if found in PATH, else raise.

    Args:
        name: Executable name (e.g., 'pw.x' or 'vasp_std').

    Returns:
        The absolute path to the executable.
    """
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(f"Executable '{name}' not found in PATH")


@tool
def validate_bins(require_qe: bool = False, require_vasp: bool = False) -> str:
    """
    Validate that required DFT binaries are available via env vars or PATH.

    Args:
        require_qe: If True, ensure $QE_BIN exists or 'pw.x' is discoverable.
        require_vasp: If True, ensure $VASP_BIN exists or 'vasp_std' is discoverable.

    Returns:
        A short summary string of discovered binaries.
    """
    msgs: List[str] = []
    if require_qe:
        qe = os.environ.get("QE_BIN", "pw.x")
        try:
            path = which_bin(qe if os.path.basename(qe) == qe else os.path.basename(qe))
            msgs.append(f"QE OK: {path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                "QE binary not found. Set QE_BIN env var or add 'pw.x' to PATH."
            )
    if require_vasp:
        vb = os.environ.get("VASP_BIN", "vasp_std")
        try:
            path = which_bin(vb if os.path.basename(vb) == vb else os.path.basename(vb))
            msgs.append(f"VASP OK: {path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                "VASP binary not found. Set VASP_BIN env var or add 'vasp_std' to PATH."
            )
    return "; ".join(msgs) if msgs else "No binaries requested for validation."
