"""Artifact installers — one per artifact type."""

from .agents import AgentInstaller
from .bundles import BundleInstaller
from .datasets import DatasetInstaller
from .skills import SkillsInstaller
from .templates import TemplateInstaller
from .tools import ToolInstaller

__all__ = [
    "AgentInstaller",
    "BundleInstaller",
    "DatasetInstaller",
    "SkillsInstaller",
    "TemplateInstaller",
    "ToolInstaller",
]
