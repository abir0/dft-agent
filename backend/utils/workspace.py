"""
Workspace Management Utilities for DFT Agent

Handles creation and management of user-specific workspaces for DFT calculations.
"""

import asyncio
import shutil
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
        workspace_path = self.base_dir / thread_id
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        self._create_standard_subdirs(workspace_path)

        return workspace_path

    def _create_standard_subdirs(self, workspace_path: Path):
        """Create standard subdirectories in the workspace.

        Args:
            workspace_path: Path to the workspace directory
        """
        dirs = [
            "structures",  # All structure files (bulk, slabs, supercells, etc.)
            "calculations",  # All calculation inputs and outputs
            "results",  # Final results and analysis
            "databases",  # Local calculation databases
        ]

        for subdir in dirs:
            (workspace_path / subdir).mkdir(parents=True, exist_ok=True)

    def get_subdir_path(self, thread_id: Optional[str], subdir: str) -> Path:
        """Get path to a specific subdirectory within a workspace.

        Args:
            thread_id: Thread/chat identifier
            subdir: Subdirectory path (e.g., 'structures', 'calculations')

        Returns:
            Path to the specified subdirectory
        """
        if not thread_id:
            thread_id = "default"
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
        workspace_path = self.base_dir / thread_id

        if workspace_path.exists():
            shutil.rmtree(workspace_path)
            return True
        return False


# Global workspace manager instance
workspace_manager = WorkspaceManager()


def get_workspace_path(thread_id: Optional[str]) -> Path:
    """Convenience function to get workspace path for a thread.

    Args:
        thread_id: Thread/chat identifier

    Returns:
        Path to the thread-specific workspace
    """
    return workspace_manager.get_workspace_path(thread_id)


async def async_get_workspace_path(thread_id: str) -> Path:
    """Asynchronously obtain (and create) the workspace path for a thread.

    Offloads the potentially blocking directory creation to a worker thread
    to keep the event loop responsive under LangGraph dev's blocking detector.
    """
    return await asyncio.to_thread(workspace_manager.get_workspace_path, thread_id)


def get_subdir_path(thread_id: str, subdir: str) -> Path:
    """Convenience function to get subdirectory path within a workspace.

    Args:
        thread_id: Thread/chat identifier
        subdir: Subdirectory name

    Returns:
        Path to the specified subdirectory
    """
    return workspace_manager.get_subdir_path(thread_id, subdir)
