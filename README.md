# Hermes OpenAI Codex Web Provider

Standalone, search-only web provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It calls OpenAI's native Responses `web_search` tool using the ChatGPT/Codex OAuth credentials already managed by Hermes. It does not require the Codex CLI or a separate search API key.

## Requirements

- Hermes Agent 0.20.6 or newer
- A ChatGPT account with Codex access
- A Codex model ID accepted for that account

## Install

Install from GitHub and enable the plugin:

```bash
hermes plugins install mazzanfar/hermes-openai-codex-web-provider --enable
```

Pip installation into the same Python environment as Hermes:

```bash
pip install git+https://github.com/mazzanfar/hermes-openai-codex-web-provider.git
hermes plugins enable openai-codex-web
```

Authenticate with the existing Hermes Codex provider:

```bash
hermes auth add openai-codex --type oauth
```

## Configure

Add the plugin settings and select its explicit backend ID in `~/.hermes/config.yaml`:

```yaml
plugins:
  entries:
    openai-codex-web:
      settings:
        model: gpt-5.6
        timeout: 90

web:
  search_backend: openai-codex-web
```

`model` is required and intentionally has no default because accepted model IDs
depend on the ChatGPT account. `timeout` defaults to 90 seconds.

The provider supports search only. Configure a separate `web.extract_backend`
if you also use `web_extract`.

## Development

Run the native plugin validator and focused tests:

```bash
hermes plugins doctor . --ci
python -m pytest
```

