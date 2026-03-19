"""Install subsystem — manages skill packs, templates, and IDE targets."""

from .registry import ContentItem, Pack, list_packs, load_pack
from .targets import detect_targets, get_all_targets, get_target

__all__ = [
    "ContentItem",
    "Pack",
    "list_packs",
    "load_pack",
    "detect_targets",
    "get_all_targets",
    "get_target",
]
