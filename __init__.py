"""Directory-install entry point for the Hermes OpenAI Codex web plugin."""

if __package__:
    from .hermes_openai_codex_web_provider import CodexWebSearchProvider, register
else:
    from hermes_openai_codex_web_provider import CodexWebSearchProvider, register

__all__ = ["CodexWebSearchProvider", "register"]
