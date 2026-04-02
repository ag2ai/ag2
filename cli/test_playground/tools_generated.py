"""Auto-generated AG2 tools by `ag2 proxy`."""

from __future__ import annotations

import httpx


def list_tasks(status: str | None = None, limit: int | None = None) -> str:
    """List all tasks"""
    url = "https://api.example.com/v1/tasks"
    params = {"status": status, "limit": limit}
    params = {k: v for k, v in params.items() if v is not None}
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def create_task(title: str, description: str | None = None, priority: str | None = None) -> str:
    """Create a new task"""
    url = "https://api.example.com/v1/tasks"
    body = {"title": title, "description": description, "priority": priority}
    resp = httpx.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.text


def get_task(task_id: int) -> str:
    """Get a task by ID"""
    url = f"https://api.example.com/v1/tasks/{task_id}"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def update_task(task_id: int, title: str | None = None, status: str | None = None, priority: str | None = None) -> str:
    """Update a task"""
    url = f"https://api.example.com/v1/tasks/{task_id}"
    body = {"title": title, "status": status, "priority": priority}
    resp = httpx.put(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.text


def delete_task(task_id: int) -> str:
    """Delete a task"""
    url = f"https://api.example.com/v1/tasks/{task_id}"
    resp = httpx.delete(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def list_comments(task_id: int) -> str:
    """List comments on a task"""
    url = f"https://api.example.com/v1/tasks/{task_id}/comments"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def add_comment(task_id: int, text: str) -> str:
    """Add a comment to a task"""
    url = f"https://api.example.com/v1/tasks/{task_id}/comments"
    body = {"text": text}
    resp = httpx.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.text
