"""Hermes Agent plugin registration for OpenAI Codex web search."""

from __future__ import annotations

from typing import Any

from .provider import CodexWebSearchProvider

__all__ = ["CodexWebSearchProvider", "register"]


def register(ctx: Any) -> None:
    """Register the standalone search-only provider with Hermes."""
    ctx.register_web_search_provider(
        CodexWebSearchProvider(
            config_getter=ctx.get_config,
            plugin_id=ctx.plugin_id,
        )
    )
