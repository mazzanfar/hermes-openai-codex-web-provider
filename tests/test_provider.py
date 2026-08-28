"""Behavior tests for the standalone Codex web-search provider."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _configured_codex_model(monkeypatch):
    """Keep behavior tests independent of a developer's local config."""
    from hermes_openai_codex_web_provider import provider as codex_provider

    monkeypatch.setattr(
        codex_provider,
        "_load_codex_web_config",
        lambda: {"model": "gpt-5.4-mini"},
    )


def _response_payload(text: str, *, annotations=None, sources=None) -> dict:
    output = [
        {
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": annotations or [],
            }],
        }
    ]
    if sources is not None:
        output.insert(0, {
            "type": "web_search_call",
            "action": {"sources": sources},
        })
    return {"status": "completed", "output": output}


class _FakeStreamResponse:
    def __init__(self, payload: dict, status_code: int = 200, body: str = ""):
        self.status_code = status_code
        self._payload = payload
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def text(self):
        return self._body

    def read(self):
        return self._body.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
            response = httpx.Response(self.status_code, request=request, text=self._body)
            raise httpx.HTTPStatusError("failed", request=request, response=response)

    def iter_lines(self):
        for item in self._payload.get("output", []):
            event = {
                "type": "response.output_item.done",
                "item": item,
            }
            yield "event: response.output_item.done"
            yield "data: " + json.dumps(event)
            yield ""
        terminal = {
            "type": "response.completed",
            "response": {"status": "completed", "output": None},
        }
        yield "event: response.completed"
        yield "data: " + json.dumps(terminal)
        yield ""


class _FakeClient:
    responses = []
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url, **kwargs):
        type(self).calls.append({"method": method, "url": url, **kwargs, "client": self.kwargs})
        return type(self).responses.pop(0)


def test_provider_identity_and_setup_schema():
    from hermes_openai_codex_web_provider import CodexWebSearchProvider

    provider = CodexWebSearchProvider()
    assert provider.name == "openai-codex-web"
    assert provider.supports_search() is True
    assert provider.supports_extract() is False
    assert "post_setup" not in provider.get_setup_schema()


def test_availability_reads_singleton_without_runtime_resolution(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider
    monkeypatch.setattr(codex_provider, "_is_explicitly_selected", lambda: True)

    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        lambda provider: {"tokens": {"access_token": "stored-token"}},
    )
    with patch(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        side_effect=AssertionError("availability must not refresh credentials"),
    ):
        assert codex_provider.CodexWebSearchProvider().is_available() is True


def test_availability_reads_pool_when_singleton_missing(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider
    monkeypatch.setattr(codex_provider, "_is_explicitly_selected", lambda: True)

    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: None)
    monkeypatch.setattr(
        "hermes_cli.auth.read_credential_pool",
        lambda provider: [{"access_token": "pool-token"}],
    )
    assert codex_provider.CodexWebSearchProvider().is_available() is True

def test_availability_requires_explicit_backend_selection(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    monkeypatch.setattr(codex_provider, "_is_explicitly_selected", lambda: False)
    monkeypatch.setattr(
        codex_provider,
        "_has_codex_credentials",
        lambda: pytest.fail("unselected provider must not probe credentials"),
    )

    assert codex_provider.CodexWebSearchProvider().is_available() is False


def test_search_normalizes_json_results(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    raw = json.dumps({
        "results": [
            {
                "title": "Original",
                "url": "https://example.com/original",
                "description": "Primary source.",
            },
            {
                "title": "Duplicate",
                "url": "https://example.com/original",
                "description": "Dropped.",
            },
            {"title": "Unsafe", "url": "javascript:alert(1)"},
        ]
    })
    monkeypatch.setattr(
        codex_provider.CodexWebSearchProvider,
        "_request",
        classmethod(lambda cls, *args, **kwargs: _response_payload(
            raw,
            sources=[{
                "type": "url",
                "url": "https://example.com/original",
                "title": "Original source",
            }],
        )),
    )

    result = codex_provider.CodexWebSearchProvider().search("official documentation", limit=5)

    assert result == {
        "success": True,
        "data": {"web": [{
            "title": "Original",
            "url": "https://example.com/original",
            "description": "Primary source.",
            "position": 1,
        }]},
    }


def test_search_falls_back_to_citations_and_sources(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    payload = _response_payload(
        "Search completed without JSON.",
        annotations=[{
            "type": "url_citation",
            "url": "https://docs.example/guide",
            "title": "Official guide",
        }],
        sources=[
            {
                "type": "url",
                "url": "https://docs.example/guide",
                "title": "Duplicate source",
            },
            {
                "type": "url",
                "url": "https://project.example/reference",
                "title": "Project reference",
            },
        ],
    )
    monkeypatch.setattr(
        codex_provider.CodexWebSearchProvider,
        "_request",
        classmethod(lambda cls, *args, **kwargs: payload),
    )

    result = codex_provider.CodexWebSearchProvider().search("project API", limit=5)

    assert [row["url"] for row in result["data"]["web"]] == [
        "https://docs.example/guide",
        "https://project.example/reference",
    ]
    assert [row["position"] for row in result["data"]["web"]] == [1, 2]


def test_search_rejects_model_url_not_grounded_in_native_sources(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    raw = json.dumps({
        "results": [{
            "title": "Plausible but invented",
            "url": "https://docs.example/item:42",
            "description": "The model invented this path.",
        }]
    })
    payload = _response_payload(
        raw,
        sources=[{
            "type": "url",
            "url": "https://search.example/result",
            "title": "Cited search result",
        }],
    )
    monkeypatch.setattr(
        codex_provider.CodexWebSearchProvider,
        "_request",
        classmethod(lambda cls, *args, **kwargs: payload),
    )

    result = codex_provider.CodexWebSearchProvider().search("exact document", limit=5)

    assert result["data"]["web"] == [{
        "title": "Cited search result",
        "url": "https://search.example/result",
        "description": "",
        "position": 1,
    }]


def test_grounded_result_returns_exact_native_source_url():
    from hermes_openai_codex_web_provider import CodexWebSearchProvider

    raw = json.dumps({
        "results": [{
            "title": "Original document",
            "url": "https://docs.example/item#model-fragment",
            "description": "Canonical URL.",
        }]
    })
    payload = _response_payload(
        raw,
        sources=[{
            "type": "url",
            "url": "https://docs.example/item",
        }],
    )

    results = CodexWebSearchProvider._extract_results(payload, limit=5)

    assert results[0]["url"] == "https://docs.example/item"


def test_encoded_reserved_path_is_not_equivalent_to_another_resource():
    from hermes_openai_codex_web_provider import CodexWebSearchProvider

    raw = json.dumps({
        "results": [{
            "title": "Different resource",
            "url": "https://docs.example/admin/settings",
            "description": "Must not inherit another URL's citation.",
        }]
    })
    payload = _response_payload(
        raw,
        sources=[{
            "type": "url",
            "url": "https://docs.example/admin%2Fsettings",
            "title": "Cited resource",
        }],
    )

    results = CodexWebSearchProvider._extract_results(payload, limit=5)

    assert results == [{
        "title": "Cited resource",
        "url": "https://docs.example/admin%2Fsettings",
        "description": "",
        "position": 1,
    }]


def test_search_is_one_bounded_provider_request(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    payload = _response_payload(
        json.dumps({
            "results": [{
                "title": "Right title, invented ID",
                "url": "https://docs.example/item:42",
                "description": "Must not escape the provider.",
            }]
        }),
        sources=[{
            "type": "url",
            "url": "https://search.example/result",
        }],
    )
    calls = []

    def request(cls, query, **kwargs):
        calls.append(query)
        return payload

    monkeypatch.setattr(
        codex_provider.CodexWebSearchProvider,
        "_request",
        classmethod(request),
    )

    result = codex_provider.CodexWebSearchProvider().search("exact title", limit=5)

    assert calls == ["exact title"]
    assert result["data"]["web"] == [{
        "title": "",
        "url": "https://search.example/result",
        "description": "",
        "position": 1,
    }]


def test_search_fails_when_native_web_search_was_not_invoked(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    payload = _response_payload(json.dumps({"results": []}))
    monkeypatch.setattr(
        codex_provider.CodexWebSearchProvider,
        "_request",
        classmethod(lambda cls, *args, **kwargs: payload),
    )

    result = codex_provider.CodexWebSearchProvider().search("current facts", limit=5)

    assert result["success"] is False
    assert "without invoking web search" in result["error"]


def test_multiple_native_search_actions_contribute_grounded_sources():
    from hermes_openai_codex_web_provider import CodexWebSearchProvider

    payload = {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "queries": ["broad query", "site:docs.example exact query"],
                    "sources": [{
                        "type": "url",
                        "url": "https://index.example/result",
                        "title": "Broad result",
                    }],
                },
            },
            {
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "query": "site:docs.example exact query",
                    "sources": [{
                        "type": "url",
                        "url": "https://docs.example/original",
                        "title": "Original source",
                    }],
                },
            },
            {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "not-json",
                    "annotations": [],
                }],
            },
        ],
    }

    results = CodexWebSearchProvider._extract_results(payload, limit=5)

    assert [row["url"] for row in results] == [
        "https://index.example/result",
        "https://docs.example/original",
    ]


def test_request_uses_codex_responses_native_search(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    payload = _response_payload(json.dumps({"results": []}))
    _FakeClient.calls = []
    _FakeClient.responses = [_FakeStreamResponse(payload)]
    monkeypatch.setattr("httpx.Client", _FakeClient)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda **kwargs: {
            "api_key": "oauth-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )

    result = codex_provider.CodexWebSearchProvider._request(
        "official project documentation",
        limit=5,
        model="gpt-5.4-mini",
        timeout_seconds=90,
    )

    assert result["status"] == "completed"
    call = _FakeClient.calls[0]
    assert call["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert call["json"]["tools"] == [{"type": "web_search"}]
    assert call["json"]["include"] == ["web_search_call.action.sources"]
    assert call["json"]["stream"] is True
    assert call["client"]["headers"]["Authorization"] == "Bearer oauth-token"


def test_request_refreshes_once_after_401(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    payload = _response_payload(json.dumps({"results": []}))
    _FakeClient.calls = []
    _FakeClient.responses = [
        _FakeStreamResponse({}, status_code=401),
        _FakeStreamResponse(payload),
    ]
    monkeypatch.setattr("httpx.Client", _FakeClient)
    refresh_flags = []

    def resolve(**kwargs):
        refresh_flags.append(kwargs.get("force_refresh"))
        token = "fresh-token" if kwargs.get("force_refresh") else "stale-token"
        return {
            "api_key": token,
            "base_url": "https://chatgpt.com/backend-api/codex",
        }

    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", resolve)

    codex_provider.CodexWebSearchProvider._request(
        "query", limit=5, model="gpt-5.4-mini", timeout_seconds=90,
    )

    assert refresh_flags == [False, True]
    assert _FakeClient.calls[1]["client"]["headers"]["Authorization"] == "Bearer fresh-token"


def test_incomplete_stream_is_a_failure():
    from hermes_openai_codex_web_provider import provider as codex_provider

    class Response:
        @staticmethod
        def iter_lines():
            payload = {
                "type": "response.incomplete",
                "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            }
            yield "data: " + json.dumps(payload)
            yield ""

    try:
        codex_provider._stream_response_payload(Response())
    except RuntimeError as exc:
        assert "max_output_tokens" in str(exc)
    else:
        raise AssertionError("incomplete Codex streams must not report success")


def test_truncated_stream_is_a_failure():
    from hermes_openai_codex_web_provider import provider as codex_provider

    class Response:
        @staticmethod
        def iter_lines():
            item = {
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": json.dumps({"results": []}),
                    "annotations": [],
                }],
            }
            yield "data: " + json.dumps({
                "type": "response.output_item.done",
                "item": item,
            })
            yield ""

    with pytest.raises(RuntimeError, match="before response.completed"):
        codex_provider._stream_response_payload(Response())


def test_configured_model_strips_hermes_context_suffix(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    monkeypatch.setattr(
        codex_provider,
        "_load_codex_web_config",
        lambda: {"model": "gpt-5.4-900k"},
    )
    assert codex_provider._resolve_model() == "gpt-5.4"


def test_search_requires_explicit_plugin_model(monkeypatch):
    from hermes_openai_codex_web_provider import provider as codex_provider

    monkeypatch.setattr(codex_provider, "_load_codex_web_config", lambda: {})
    monkeypatch.setattr(
        codex_provider.CodexWebSearchProvider,
        "_request",
        classmethod(lambda cls, *args, **kwargs: pytest.fail("request must not run")),
    )

    result = codex_provider.CodexWebSearchProvider().search("query", limit=5)

    assert result["success"] is False
    assert "plugins.entries.openai-codex-web.settings.model" in result["error"]


def test_provider_uses_explicit_standalone_backend_id():
    from hermes_openai_codex_web_provider import CodexWebSearchProvider

    assert CodexWebSearchProvider().name == "openai-codex-web"


def test_register_adds_provider_to_context():
    from hermes_openai_codex_web_provider import register

    registered = []

    class Context:
        plugin_id = "directory-install-id"

        @staticmethod
        def get_config(key, default=None):
            return {"model": "gpt-test", "timeout": 45}.get(key, default)

        @staticmethod
        def register_web_search_provider(provider):
            registered.append(provider)

    register(Context())

    assert len(registered) == 1
    assert registered[0].name == "openai-codex-web"
    assert registered[0]._load_config() == {"model": "gpt-test", "timeout": 45}
    assert registered[0]._plugin_id == "directory-install-id"
