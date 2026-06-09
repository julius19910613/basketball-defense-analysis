from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from uuid import uuid4

from app.analysis.schemas import AnalysisResponse


class TaskState:
    """Represents the in-memory state of a background analysis task."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "pending"  # pending | processing | completed | failed
        self.progress = 0
        self.result: Optional[AnalysisResponse] = None
        self.error: Optional[str] = None


class TaskManager:
    """Thread-safe manager for background analysis task states."""

    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        """Create a new task in pending state and return its task_id."""
        task_id = str(uuid4().hex)
        state = TaskState(task_id)
        with self._lock:
            self._tasks[task_id] = state
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Retrieve the state of a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: str, progress: int = None, error: str = None) -> None:
        """Update the status, progress, or error of a task."""
        with self._lock:
            state = self._tasks.get(task_id)
            if state:
                state.status = status
                if progress is not None:
                    state.progress = progress
                if error is not None:
                    state.error = error

    def set_result(self, task_id: str, result: AnalysisResponse) -> None:
        """Complete a task by storing its final result payload."""
        with self._lock:
            state = self._tasks.get(task_id)
            if state:
                state.status = "completed"
                state.progress = 100
                state.result = result


# Global singleton instance of TaskManager
_global_task_manager = TaskManager()

def get_task_manager() -> TaskManager:
    """Return the global TaskManager instance."""
    return _global_task_manager
