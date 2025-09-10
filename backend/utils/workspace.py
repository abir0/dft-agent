"""
Workspace Management Utilities for DFT Agent

Handles creation and management of user-specific workspaces for DFT calculations.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Optional

from backend.settings import settings


class WorkspaceManager:
    """Manages user-specific workspaces for DFT calculations."""

    def __init__(self, base_workspace_dir: Optional[str] = None):
        """Initialize workspace manager.

        Args:
            base_workspace_dir: Base directory for all workspaces.
                               Defaults to ROOT_PATH/WORKSPACE
        """
        if base_workspace_dir is None:
            self.base_dir = Path(settings.ROOT_PATH) / "WORKSPACE"
        else:
            self.base_dir = Path(base_workspace_dir)

        # Ensure base workspace directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_workspace_path(self, thread_id: str) -> Path:
        """Get the workspace path for a specific thread/chat ID.

        Args:
            thread_id: Unique thread/chat identifier

        Returns:
            Path to the thread-specific workspace directory
        """
        # Sanitize thread_id to ensure it's filesystem-safe
        safe_thread_id = self._sanitize_thread_id(thread_id)
        workspace_path = self.base_dir / safe_thread_id

        # Create workspace directory if it doesn't exist
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        self._create_standard_subdirs(workspace_path)

        return workspace_path

    def _sanitize_thread_id(self, thread_id: str) -> str:
        """Sanitize thread ID to be filesystem-safe.

        Args:
            thread_id: Original thread ID

        Returns:
            Filesystem-safe thread ID
        """
        # If thread_id is None or empty, generate a random one
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Create a hash if the thread_id is too long or contains unsafe characters
        if (
            len(thread_id) > 50
            or not thread_id.replace("-", "").replace("_", "").isalnum()
        ):
            # Use first 8 chars + hash for readability
            prefix = "".join(c for c in thread_id[:8] if c.isalnum())
            hash_suffix = hashlib.md5(thread_id.encode()).hexdigest()[:8]
            return f"{prefix}_{hash_suffix}" if prefix else f"ws_{hash_suffix}"

        return thread_id

    def _create_standard_subdirs(self, workspace_path: Path):
        """Create standard subdirectories in the workspace.

        Args:
            workspace_path: Path to the workspace directory
        """
        standard_dirs = [
            "structures",
            "structures/bulk",
            "structures/supercells",
            "structures/slabs",
            "structures/with_adsorbates",
            "calculations",
            "calculations/qe_inputs",
            "calculations/jobs",
            "calculations/outputs",
            "optimized",
            "relaxed",
            "kpaths",
            "kpoints",
            "convergence_tests",
            "results",
            "databases",
        ]

        for subdir in standard_dirs:
            (workspace_path / subdir).mkdir(parents=True, exist_ok=True)

    def get_subdir_path(self, thread_id: str, subdir: str) -> Path:
        """Get path to a specific subdirectory within a workspace.

        Args:
            thread_id: Thread/chat identifier
            subdir: Subdirectory name (e.g., 'structures/bulk', 'calculations')

        Returns:
            Path to the specified subdirectory
        """
        workspace_path = self.get_workspace_path(thread_id)
        subdir_path = workspace_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        return subdir_path

    def list_workspaces(self) -> list[str]:
        """List all existing workspace IDs.

        Returns:
            List of workspace thread IDs
        """
        if not self.base_dir.exists():
            return []

        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def cleanup_workspace(self, thread_id: str) -> bool:
        """Remove a workspace and all its contents.

        Args:
            thread_id: Thread/chat identifier

        Returns:
            True if workspace was removed, False if it didn't exist
        """
        workspace_path = self.base_dir / self._sanitize_thread_id(thread_id)

        if workspace_path.exists():
            import shutil

            shutil.rmtree(workspace_path)
            return True
        return False

    def get_workspace_info(self, thread_id: str) -> dict:
        """Get information about a workspace.

        Args:
            thread_id: Thread/chat identifier

        Returns:
            Dictionary with workspace information
        """
        workspace_path = self.get_workspace_path(thread_id)

        # Count files in different categories
        structure_files = len(list(workspace_path.glob("structures/**/*.*")))
        calculation_files = len(list(workspace_path.glob("calculations/**/*.*")))
        result_files = len(list(workspace_path.glob("results/**/*.*")))

        # Get workspace size
        total_size = sum(
            f.stat().st_size for f in workspace_path.rglob("*") if f.is_file()
        )

        return {
            "thread_id": thread_id,
            "workspace_path": str(workspace_path),
            "structure_files": structure_files,
            "calculation_files": calculation_files,
            "result_files": result_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "created": workspace_path.stat().st_ctime
            if workspace_path.exists()
            else None,
        }


# Global workspace manager instance
workspace_manager = WorkspaceManager()


def get_workspace_path(thread_id: str) -> Path:
    """Convenience function to get workspace path for a thread.

    Args:
        thread_id: Thread/chat identifier

    Returns:
        Path to the thread-specific workspace
    """
    return workspace_manager.get_workspace_path(thread_id)


def get_subdir_path(thread_id: str, subdir: str) -> Path:
    """Convenience function to get subdirectory path within a workspace.

    Args:
        thread_id: Thread/chat identifier
        subdir: Subdirectory name

    Returns:
        Path to the specified subdirectory
    """
    return workspace_manager.get_subdir_path(thread_id, subdir)


def extract_thread_id_from_config(config: Optional[dict] = None) -> str:
    """Extract thread ID from LangGraph config.

    Args:
        config: LangGraph configuration dictionary

    Returns:
        Thread ID string, or generates a new one if not found
    """
    if config is None:
        return str(uuid.uuid4())

    # Try to get thread_id from various possible locations in config
    configurable = config.get("configurable", {})

    # Common locations for thread_id in LangGraph configs
    thread_id = (
        configurable.get("thread_id")
        or configurable.get("session_id")
        or configurable.get("conversation_id")
        or config.get("thread_id")
        or config.get("session_id")
        or config.get("conversation_id")
    )

    if thread_id:
        return str(thread_id)

    # If no thread_id found, generate a new one
    return str(uuid.uuid4())
