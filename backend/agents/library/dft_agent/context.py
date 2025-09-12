"""
Context management for DFT agent workspace.

Provides workspace context to tools so they can access the correct workspace directory.
"""

from typing import Optional

# Context variable to store current thread_id
_current_thread_id: Optional[str] = None


def set_current_thread_id(thread_id: str):
    """Set the current thread ID for tool execution context."""
    global _current_thread_id
    _current_thread_id = thread_id


def get_current_thread_id() -> Optional[str]:
    """Get the current thread ID from execution context."""
    return _current_thread_id


def clear_thread_context():
    """Clear the current thread context."""
    global _current_thread_id
    _current_thread_id = None
