"""Standalone OpenAI Codex web-search provider for Hermes Agent.

The plugin runs a bounded Responses request with OpenAI's native
``web_search`` tool and Hermes-managed ``openai-codex`` OAuth credentials.
It registers the explicit-only ``openai-codex-web`` search backend and returns
Hermes's standard search-result envelope.

Plugin settings live under
``plugins.entries.openai-codex-web.settings``:

    model: <eligible Codex model ID>
    timeout: 90

Select it with ``web.search_backend: openai-codex-web``. The provider is
search-only; configure a separate extraction backend when needed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 90.0
PLUGIN_ID = "openai-codex-web"
_MAX_ERROR_BODY_CHARS = 500
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _load_codex_web_config() -> Dict[str, Any]:
    """Fallback settings reader for direct construction outside plugin loading."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, dict) else None
        entries = plugins.get("entries") if isinstance(plugins, dict) else None
        entry = entries.get(PLUGIN_ID) if isinstance(entries, dict) else None
        settings = entry.get("settings") if isinstance(entry, dict) else None
        return dict(settings) if isinstance(settings, dict) else {}
    except Exception as exc:  # noqa: BLE001 — config is optional here
        logger.debug("Could not load fallback plugin settings: %s", exc)
        return {}
def _resolve_model(
    config: Optional[Dict[str, Any]] = None,
    *,
    plugin_id: str = PLUGIN_ID,
) -> str:
    """Return the explicitly configured Codex search model."""
    value = (config if config is not None else _load_codex_web_config()).get("model")
    if isinstance(value, str) and value.strip():
        model = value.strip()
        from agent.model_metadata import strip_codex_context_variant_suffix

        return strip_codex_context_variant_suffix(model)
    raise RuntimeError(
        "Codex web search requires an explicit model. Set "
        f"`plugins.entries.{plugin_id}.settings.model` in config.yaml to a "
        "model available to your Codex account."
    )

def _is_explicitly_selected() -> bool:
    """Prevent Hermes's custom-provider fallback from auto-selecting Codex."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        web = config.get("web") if isinstance(config, dict) else None
        if not isinstance(web, dict):
            return False
        return any(
            str(web.get(key) or "").strip().lower() == PLUGIN_ID
            for key in ("search_backend", "backend")
        )
    except Exception as exc:  # noqa: BLE001 — unavailable is the safe default
        logger.debug("Could not read explicit web backend selection: %s", exc)
        return False






def _has_codex_credentials() -> bool:
    """Cheap local credential probe used by discovery and setup UI.

    This reads persisted state only. It never refreshes OAuth, selects a live
    pool entry, or performs network I/O; runtime search owns those operations.
    """
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("openai-codex") or {}
        tokens = state.get("tokens") if isinstance(state, dict) else None
        if isinstance(tokens, dict) and str(tokens.get("access_token") or "").strip():
            return True
    except Exception:
        pass

    try:
        from hermes_cli.auth import read_credential_pool

        entries = read_credential_pool("openai-codex")
        return any(
            isinstance(entry, dict)
            and str(entry.get("access_token") or "").strip()
            for entry in entries
        )
    except Exception:
        return False


def _resolve_codex_credentials(
    *,
    force_refresh: bool = False,
    token_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one runtime Codex credential through Hermes's pool first.

    Pool-first selection preserves multi-account rotation. On a 401, refresh
    the credential that issued the failed request rather than an unrelated
    singleton token. Legacy singleton auth remains the fallback.
    """
    try:
        from agent.credential_pool import load_pool
        from hermes_cli.auth import DEFAULT_CODEX_BASE_URL

        pool = load_pool("openai-codex")
        if pool and pool.has_credentials():
            entry = (
                pool.try_refresh_matching(api_key_hint=token_hint)
                if force_refresh
                else pool.select()
            )
            if entry is not None:
                token = str(
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                    or ""
                ).strip()
                base_url = str(
                    getattr(entry, "runtime_base_url", None)
                    or getattr(entry, "base_url", "")
                    or DEFAULT_CODEX_BASE_URL
                ).strip().rstrip("/")
                if token and base_url:
                    return {"api_key": token, "base_url": base_url, "source": "credential_pool"}
    except Exception as exc:  # noqa: BLE001 — singleton fallback below
        logger.debug("Codex web credential-pool resolution failed: %s", exc)

    from hermes_cli.auth import resolve_codex_runtime_credentials

    return resolve_codex_runtime_credentials(force_refresh=force_refresh)


def _iter_sse_json(response: Any) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from an HTTPX SSE response."""
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush() -> Optional[Dict[str, Any]]:
        nonlocal event_name, data_lines
        raw = "\n".join(data_lines).strip()
        event = event_name
        event_name = None
        data_lines = []
        if not raw or raw == "[DONE]":
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        if event and "type" not in payload:
            payload["type"] = event
        return payload

    for raw_line in response.iter_lines():
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else str(raw_line)
        )
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    payload = flush()
    if payload is not None:
        yield payload


def _stream_response_payload(response: Any) -> Dict[str, Any]:
    """Reconstruct the useful Responses payload from SSE events.

    Codex's terminal event has historically drifted between carrying a full
    ``response.output`` list and omitting it.  Prefer independently completed
    output items, falling back to the terminal response when necessary.
    """
    output_items: List[Dict[str, Any]] = []
    terminal: Dict[str, Any] = {}
    completed = False

    for event in _iter_sse_json(response):
        event_type = str(event.get("type") or "")
        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif event_type == "response.completed":
            completed_response = event.get("response")
            if isinstance(completed_response, dict):
                terminal = dict(completed_response)
            completed = True
        elif event_type == "response.incomplete":
            response_obj = event.get("response")
            details = (
                response_obj.get("incomplete_details")
                if isinstance(response_obj, dict)
                else None
            )
            reason = details.get("reason") if isinstance(details, dict) else None
            raise RuntimeError(
                f"Codex web search response was incomplete: {reason or 'unknown reason'}"
            )
        elif event_type in {"response.failed", "error"}:
            error = event.get("error")
            if not isinstance(error, dict):
                response_obj = event.get("response")
                error = response_obj.get("error") if isinstance(response_obj, dict) else None
            message = (
                error.get("message")
                if isinstance(error, dict)
                else event.get("message")
            )
            raise RuntimeError(str(message or "Codex web search stream failed"))

    if not completed:
        raise RuntimeError(
            "Codex web search stream ended before response.completed"
        )
    if output_items:
        terminal["output"] = output_items
    elif not isinstance(terminal.get("output"), list):
        terminal["output"] = []
    return terminal


def _summarize_error_body(body: str) -> str:
    """Return a bounded, useful HTTP error description."""
    try:
        payload = json.loads(body or "")
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message.strip():
            return message.strip()[:_MAX_ERROR_BODY_CHARS]
    except (TypeError, ValueError):
        pass
    return (body or "")[:_MAX_ERROR_BODY_CHARS]


def _valid_web_url(value: Any) -> str:
    """Return a normalized absolute HTTP(S) URL, or an empty string."""
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _url_key(value: Any) -> str:
    """Canonical comparison key that preserves URL resource semantics."""
    url = _valid_web_url(value)
    if not url:
        return ""
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


class CodexWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by OpenAI Codex native web search."""
    def __init__(
        self,
        *,
        config_getter: Optional[Callable[..., Any]] = None,
        plugin_id: str = PLUGIN_ID,
    ) -> None:
        self._config_getter = config_getter
        self._plugin_id = plugin_id

    def _load_config(self) -> Dict[str, Any]:
        if self._config_getter is None:
            return _load_codex_web_config()
        return {
            "model": self._config_getter("model"),
            "timeout": self._config_getter("timeout", DEFAULT_TIMEOUT),
        }


    @property
    def name(self) -> str:
        return PLUGIN_ID

    @property
    def display_name(self) -> str:
        return "OpenAI Codex Web Search"

    def is_available(self) -> bool:
        return _is_explicitly_selected() and _has_codex_credentials()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Run one bounded Codex search turn and normalize its results."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:
            pass

        try:
            safe_limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            safe_limit = 5

        try:
            cfg = self._load_config()
            model = _resolve_model(cfg, plugin_id=self._plugin_id)
            timeout_seconds = max(10.0, float(cfg.get("timeout", DEFAULT_TIMEOUT)))
        except (TypeError, ValueError):
            timeout_seconds = DEFAULT_TIMEOUT
        except RuntimeError as exc:
            return {"success": False, "error": str(exc)}

        try:
            response_data = self._request(
                str(query),
                limit=safe_limit,
                model=model,
                timeout_seconds=timeout_seconds,
            )
            if not self._has_native_search_call(response_data):
                raise RuntimeError(
                    "Codex hosted search completed without invoking web search"
                )
        except Exception as exc:  # noqa: BLE001 — provider boundary returns envelopes
            logger.warning("Codex web search failed: %s", exc)
            return {"success": False, "error": str(exc)}

        return {
            "success": True,
            "data": {"web": self._extract_results(response_data, limit=safe_limit)},
        }

    @classmethod
    def _request(
        cls,
        query: str,
        *,
        limit: int,
        model: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        """Send one streaming Responses request, refreshing once on HTTP 401."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is not installed (required for Codex web search)") from exc

        from agent.auxiliary_client import _codex_cloudflare_headers
        payload: Dict[str, Any] = {
            "model": model,
            "store": False,
            "instructions": (
                "You are Hermes's bounded web-search worker. Use the native "
                "web_search tool to research the user's query, refining or "
                "cross-checking with additional searches inside this response when "
                "useful. Prefer original, authoritative sources and direct canonical "
                "URLs. Every returned URL must be copied from a source web_search "
                "actually visited or cited; never invent, infer, or reconstruct a "
                "URL. Return only the requested JSON object."
            ),
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": cls._build_prompt(query, limit),
                }],
            }],
            "tools": [{"type": "web_search"}],
            "include": ["web_search_call.action.sources"],
            "stream": True,
        }
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=min(30.0, timeout_seconds),
            read=timeout_seconds,
            write=min(30.0, timeout_seconds),
            pool=min(30.0, timeout_seconds),
        )

        for attempt in range(2):
            try:
                creds = _resolve_codex_credentials(
                    force_refresh=attempt == 1,
                    token_hint=token if attempt == 1 else None,
                )
            except Exception as exc:  # noqa: BLE001 — convert auth errors at boundary
                raise RuntimeError(
                    "No usable Codex/ChatGPT OAuth credentials. Run "
                    "`hermes auth add openai-codex --type oauth`."
                ) from exc

            token = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip().rstrip("/")
            if not token or not base_url:
                raise RuntimeError(
                    "No usable Codex/ChatGPT OAuth credentials. Run "
                    "`hermes auth add openai-codex --type oauth`."
                )

            headers = _codex_cloudflare_headers(token, base_url=base_url)
            headers.update({
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            })

            logger.info("Codex web search: %r (limit=%d, model=%s)", query, limit, model)
            try:
                with httpx.Client(timeout=timeout, headers=headers) as client:
                    with client.stream(
                        "POST", f"{base_url}/responses", json=payload,
                    ) as response:
                        if response.status_code == 401 and attempt == 0:
                            response.read()
                            continue
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            response.read()
                            detail = _summarize_error_body(response.text)
                            if response.status_code == 429:
                                raise RuntimeError(
                                    "Codex web search is rate-limited or the ChatGPT "
                                    f"usage limit is exhausted: {detail}"
                                ) from exc
                            raise RuntimeError(
                                f"Codex web search returned HTTP {response.status_code}: "
                                f"{detail}"
                            ) from exc
                        return _stream_response_payload(response)
            except httpx.RequestError as exc:
                raise RuntimeError(f"Could not reach OpenAI Codex: {exc}") from exc

        raise RuntimeError(
            "Codex authentication was rejected after one token refresh. Run "
            "`hermes auth add openai-codex --type oauth` to sign in again."
        )

    @staticmethod
    def _build_prompt(query: str, limit: int) -> str:
        return (
            "Search the web for the query below. Return ONLY one JSON object "
            "matching this schema, with no prose or markdown fences:\n"
            '{"results": [{"title": "string", "url": "https://...", '
            '"description": "concise factual summary"}]}\n'
            f"Return at most {limit} results ordered by relevance. Prefer original "
            "manufacturer, author, project, registry, documentation, or repository "
            "pages over aggregators. Use only URLs present in search sources, and "
            "return an empty results array when no reliable cited result exists.\n\n"
            f"Query: {query}"
        )

    @staticmethod
    def _has_native_search_call(response_data: Dict[str, Any]) -> bool:
        output = response_data.get("output")
        return isinstance(output, list) and any(
            isinstance(item, dict) and item.get("type") == "web_search_call"
            for item in output
        )


    @classmethod
    def _extract_results(
        cls,
        response_data: Dict[str, Any],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        text_blocks, annotations, sources = cls._collect_response_parts(response_data)

        evidence = [*annotations, *sources]
        evidence_by_key: Dict[str, Dict[str, Any]] = {}
        for item in evidence:
            key = _url_key(item.get("url"))
            if key and key not in evidence_by_key:
                evidence_by_key[key] = item

        for text in text_blocks:
            parsed = cls._try_parse_json_results(text, limit=limit)
            if parsed is not None:
                # URLs are security- and correctness-sensitive data. The model
                # may synthesize a plausible but nonexistent item ID even when
                # its title is right. Accept only URLs grounded in the native
                # web_search call's citations/sources; use model text solely to
                # enrich those grounded URLs.
                grounded: List[Dict[str, Any]] = []
                seen_grounded: set[str] = set()
                for row in parsed:
                    key = _url_key(row.get("url"))
                    if not key or key not in evidence_by_key or key in seen_grounded:
                        continue
                    seen_grounded.add(key)
                    grounded_row = dict(row)
                    grounded_row["url"] = _valid_web_url(
                        evidence_by_key[key].get("url")
                    )
                    grounded_row["position"] = len(grounded) + 1
                    grounded.append(grounded_row)
                    if len(grounded) >= limit:
                        return grounded
                if grounded:
                    return grounded

        candidates: List[Dict[str, Any]] = evidence
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            url = _valid_web_url(candidate.get("url"))
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({
                "title": str(candidate.get("title") or "").strip(),
                "url": url,
                "description": "",
                "position": len(results) + 1,
            })
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _collect_response_parts(
        response_data: Dict[str, Any],
    ) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        text_blocks: List[str] = []
        annotations: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        output = response_data.get("output")
        if not isinstance(output, list):
            return text_blocks, annotations, sources

        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for chunk in item.get("content") or []:
                    if not isinstance(chunk, dict) or chunk.get("type") != "output_text":
                        continue
                    text = chunk.get("text")
                    if isinstance(text, str) and text.strip():
                        text_blocks.append(text)
                    for annotation in chunk.get("annotations") or []:
                        if isinstance(annotation, dict) and annotation.get("type") == "url_citation":
                            annotations.append(annotation)
            elif item.get("type") == "web_search_call":
                action = item.get("action")
                if isinstance(action, dict):
                    for source in action.get("sources") or []:
                        if isinstance(source, dict):
                            sources.append(source)
        return text_blocks, annotations, sources

    @staticmethod
    def _try_parse_json_results(
        text: str,
        *,
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        parsed = CodexWebSearchProvider._try_parse_json_payload(text)
        rows = parsed.get("results") if isinstance(parsed, dict) else None
        if isinstance(rows, list):
            seen: set[str] = set()
            normalized: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                url = _valid_web_url(row.get("url"))
                if not url or url in seen:
                    continue
                seen.add(url)
                normalized.append({
                    "title": str(row.get("title") or "").strip(),
                    "url": url,
                    "description": str(row.get("description") or "").strip(),
                    "position": len(normalized) + 1,
                })
                if len(normalized) >= limit:
                    break
            return normalized
        return None

    @staticmethod
    def _try_parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
        """Parse the requested JSON object, tolerating surrounding prose."""
        candidates = [text]
        match = _JSON_BLOCK_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                return parsed
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "subscription · search only",
            "tag": (
                "Agentic native web search using your ChatGPT/Codex subscription; "
                "requires an explicit Codex model ID but no separate search API key."
            ),
            "env_vars": [],
        }
