# Hermes OpenAI Codex Web Provider

Standalone, search-only web provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It calls OpenAI's native Responses `web_search` tool using the ChatGPT/Codex OAuth credentials already managed by Hermes. It does not require the Codex CLI or a separate search API key.

## Requirements

- Hermes Agent 0.20.6 or newer
- A ChatGPT account with Codex access
- A Codex model ID accepted for that account

## Install

Hermes supports both native directory plugins and pip entry-point plugins.

Native installation from GitHub:

```bash
hermes plugins install mazzanfar/hermes-openai-codex-web-provider --no-enable
hermes plugins enable openai-codex-web
```

Pip installation into the same Python environment as Hermes:

```bash
pip install git+https://github.com/mazzanfar/hermes-openai-codex-web-provider.git
hermes plugins enable openai-codex-web
```

Then authenticate the existing Hermes Codex provider:

```bash
hermes auth add openai-codex --type oauth
```
Confirm Hermes discovered and enabled the plugin:

```bash
hermes plugins list
hermes plugins doctor openai-codex-web
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
  extract_backend: firecrawl
```

`model` is required and intentionally has no default because the model IDs accepted by ChatGPT/Codex accounts change independently of this plugin. `timeout` defaults to 90 seconds and is clamped to a minimum of 10 seconds.

The provider never auto-selects itself: its backend ID is distinct from Hermes's built-in providers and must be assigned to `web.search_backend` or `web.backend`. It supports search only. Configure Firecrawl, Tavily, Exa, or Parallel separately for `web_extract`.

## Community discovery

Until this plugin is accepted into the Hermes community plugin index, users
install it with the GitHub `owner/repo` command above. After it is indexed,
users can discover and install it by name:

```bash
hermes plugins search codex
hermes plugins install openai-codex-web
```

Index inclusion reviews metadata, not plugin code. Hermes still shows its
normal source review and enablement flow during installation.

## Development

Use a Hermes checkout or environment containing `hermes-agent`, then run the
native plugin validator and focused tests:

```bash
hermes plugins doctor . --ci
python -m pytest
```

The provider returns only URLs grounded in the native search call's source or citation metadata, retries one HTTP 401 after refreshing the matching Hermes credential-pool entry, and preserves Hermes's standard web result envelope.
