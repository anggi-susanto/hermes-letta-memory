# hermes-letta-memory

Letta memory provider plugin for Hermes Agent — structured long-term memory backed by a self-hosted Letta API.

Hermes stays the primary agent; Letta acts as the memory service (one lightweight Letta agent per user: profile, preference, episodic, summary).

## Install

```bash
hermes plugins install anggi-susanto/hermes-letta-memory
```

Or as a Hermes memory provider it installs to `~/.hermes/plugins/letta/`
(`plugin.yaml` name is `letta`), then:

```bash
hermes memory setup   # choose letta
```

## Configure

`memory.provider: letta` plus:

```yaml
memory:
  letta:
    enabled: true
    base_url: https://letta.example.com
    prefetch_top_k: 5
```

Optional secrets via env: `LETTA_API_KEY`, `LETTA_PROJECT_ID`, `LETTA_USER_ID_HEADER`, `LETTA_BASE_URL`.

## Source

Extracted from Hermes Agent core (`plugins/memory/letta`) before it was dropped upstream, published as a user plugin so `memory.provider: letta` keeps working.
