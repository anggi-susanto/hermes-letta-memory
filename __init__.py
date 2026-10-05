"""Letta memory plugin — external memory provider using a self-hosted Letta API.

Mode B integration: Hermes remains the primary agent; Letta acts as a structured
memory service. The provider stores durable memory as archival-memory passages on
one lightweight Letta agent per canonical user.

Memory model:
- profile
- preference
- episodic
- summary

Write policy:
- mirror built-in memory writes (user/profile durable facts)
- persist non-transient conversation turns for continuity
- optional session-end summaries
- ignore transient turn-by-turn chat spam only when the exchange is trivial

Read policy:
- prefetch on session start / ambiguous queries / project references
- tool access for explicit inspection/search/write
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib import error, parse, request


def _prompt_text(label: str, default: Optional[str] = None, secret: bool = False) -> str:
    prompt = f"  {label}"
    if default not in (None, ""):
        prompt += f" [{default}]"
    prompt += ": "
    if secret:
        import getpass
        return getpass.getpass(prompt)
    value = input(prompt).strip()
    return value or (default or "")


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"  {label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true"}


def _write_env_vars(env_path: Path, env_writes: Dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: List[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()
    updated_keys = set()
    new_lines: List[str] = []
    for line in existing_lines:
        key, sep, _value = line.partition("=")
        stripped = key.strip()
        if sep and stripped in env_writes:
            new_lines.append(f"{stripped}={env_writes[stripped]}")
            updated_keys.add(stripped)
        else:
            new_lines.append(line)
    for key, value in env_writes.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")

from agent.memory_manager import sanitize_context
from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

PROFILE_SCHEMA = {
    "name": "letta_profile",
    "description": "Get compact structured profile/preference memory for the current user from Letta.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "letta_search",
    "description": "Semantic search over Letta archival memory for the current user. Use for cross-session recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Maximum results to return (default 5)."},
            "memory_type": {
                "type": "string",
                "description": "Optional filter: profile, preference, episodic, or summary.",
                "enum": ["profile", "preference", "episodic", "summary"],
            },
            "project": {"type": "string", "description": "Optional project namespace filter."},
        },
        "required": ["query"],
    },
}

CONTEXT_SCHEMA = {
    "name": "letta_context",
    "description": "Ask for compact relevant Letta memory context about the current user or project.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language question about remembered user/project context."},
            "project": {"type": "string", "description": "Optional project namespace filter."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "letta_conclude",
    "description": "Write a durable structured memory entry into Letta for the current user.",
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "Durable fact/preference to store."},
            "memory_type": {
                "type": "string",
                "description": "Memory type bucket.",
                "enum": ["profile", "preference", "episodic", "summary"],
                "default": "preference",
            },
            "project": {"type": "string", "description": "Optional project namespace."},
            "confidence": {"type": "number", "description": "Confidence 0..1 (default 0.9)."},
            "source": {"type": "string", "description": "Source tag (default user_explicit)."},
        },
        "required": ["conclusion"],
    },
}

ALL_TOOL_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA, CONCLUDE_SCHEMA]
ENTRY_DELIMITER = "\n§\n"


@dataclass
class LettaConfig:
    enabled: bool = True
    base_url: str = ""
    api_key: str = ""
    agent_name_prefix: str = "Hermes Memory"
    user_id_header: str = ""
    project_id: str = ""
    prefetch_top_k: int = 5
    recall_mode: str = "hybrid"
    injection_frequency: str = "every-turn"
    first_turn_context_enabled: bool = True
    context_char_limit: int = 1200
    context_tokens: int = 0
    context_cadence: int = 1
    dialectic_cadence: int = 1
    auto_session_summary: bool = True
    session_summary_char_limit: int = 1200

    @classmethod
    def from_global_config(cls) -> "LettaConfig":
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
        mem = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
        letta = mem.get("letta", {}) if isinstance(mem, dict) else {}
        env = os.environ
        return cls(
            enabled=bool(letta.get("enabled", True)),
            base_url=(env.get("LETTA_BASE_URL") or letta.get("base_url") or "").rstrip("/"),
            api_key=env.get("LETTA_API_KEY") or letta.get("api_key") or "",
            agent_name_prefix=letta.get("agent_name_prefix", "Hermes Memory"),
            user_id_header=letta.get("user_id_header", env.get("LETTA_USER_ID_HEADER", "")),
            project_id=env.get("LETTA_PROJECT_ID") or letta.get("project_id") or "",
            prefetch_top_k=int(letta.get("prefetch_top_k", 5) or 5),
            recall_mode=_normalize_recall_mode(letta.get("recall_mode") or letta.get("recallMode") or "hybrid"),
            injection_frequency=_normalize_injection_frequency(letta.get("injection_frequency") or letta.get("injectionFrequency") or "every-turn"),
            first_turn_context_enabled=_to_bool(letta.get("first_turn_context_enabled", letta.get("firstTurnContextEnabled", True)), default=True),
            context_char_limit=int(letta.get("context_char_limit", letta.get("contextCharLimit", 1200)) or 1200),
            context_tokens=int(letta.get("context_tokens", letta.get("contextTokens", 0)) or 0),
            context_cadence=max(1, int(letta.get("context_cadence", letta.get("contextCadence", 1)) or 1)),
            dialectic_cadence=max(1, int(letta.get("dialectic_cadence", letta.get("dialecticCadence", 1)) or 1)),
            auto_session_summary=str(letta.get("auto_session_summary", True)).lower() in ("true", "1", "yes"),
            session_summary_char_limit=int(letta.get("session_summary_char_limit", 1200) or 1200),
        )


def _normalize_recall_mode(value: Any) -> str:
    raw = str(value or "hybrid").strip().lower()
    if raw in {"tools", "context", "hybrid"}:
        return raw
    return "hybrid"


def _normalize_injection_frequency(value: Any) -> str:
    raw = str(value or "every-turn").strip().lower()
    if raw in {"first-turn", "every-turn"}:
        return raw
    return "every-turn"


def _to_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class LettaClient:
    def __init__(self, config: LettaConfig):
        self.config = config

    def _headers(self, *, json_body: bool = True) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "Hermes-Agent/LettaMemoryProvider"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.project_id:
            headers["X-Project-Id"] = self.config.project_id
        if self.config.user_id_header:
            headers["user_id"] = self.config.user_id_header
        return headers

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.config.base_url}{path}"
        if params:
            clean = []
            for k, v in params.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    for item in v:
                        clean.append((k, str(item)))
                else:
                    clean.append((k, str(v)))
            if clean:
                url += "?" + parse.urlencode(clean, doseq=True)
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = request.Request(url, data=data, method=method, headers=self._headers(json_body=body is not None))
        try:
            with request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{method} {path} failed: HTTP {e.code} {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"{method} {path} failed: {e.reason}") from e

    def list_agents(self, *, name: str) -> List[Dict[str, Any]]:
        data = self._request("GET", "/v1/agents/", params={"name": name, "limit": 10})
        return data if isinstance(data, list) else []

    def create_agent(self, *, name: str, tags: List[str]) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/v1/agents/",
            body={"name": name, "tags": tags, "model": "letta/letta-free"},
        )

    def get_core_memory_blocks(self, agent_id: str) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/v1/agents/{agent_id}/core-memory/blocks")
        return data if isinstance(data, list) else []

    def update_core_memory_block(self, agent_id: str, block_label: str, value: str) -> Dict[str, Any]:
        return self._request("PATCH", f"/v1/agents/{agent_id}/core-memory/blocks/{parse.quote(block_label, safe='')}", body={"value": value})

    def create_passage(self, agent_id: str, text: str) -> Dict[str, Any]:
        return self._request("POST", f"/v1/agents/{agent_id}/archival-memory", body={"text": text})

    def list_archival_memory(self, agent_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/v1/agents/{agent_id}/archival-memory", params={"limit": limit})
        return data if isinstance(data, list) else []

    def delete_archival_memory(self, agent_id: str, memory_id: str) -> Dict[str, Any]:
        data = self._request("DELETE", f"/v1/agents/{agent_id}/archival-memory/{memory_id}")
        return data if isinstance(data, dict) else {"deleted": True}

    def search_archival_memory(self, agent_id: str, *, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/v1/agents/{agent_id}/archival-memory/search", params={"query": query, "top_k": top_k})
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list):
                normalized: List[Dict[str, Any]] = []
                for item in results:
                    if isinstance(item, dict):
                        normalized.append(
                            {
                                "id": item.get("id"),
                                "text": item.get("text") or item.get("content") or "",
                                "content": item.get("content") or item.get("text") or "",
                                "timestamp": item.get("timestamp"),
                                "tags": item.get("tags") or [],
                                "_raw": item,
                            }
                        )
                return normalized
        return []


class LettaMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config: Optional[LettaConfig] = None
        self._client: Optional[LettaClient] = None
        self._session_id = ""
        self._platform = "cli"
        self._canonical_user_id = "local:default"
        self._agent_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._first_turn_context: Optional[str] = None
        self._turn_count = 0
        self._turn_sequence = 0
        self._last_context_turn = -999
        self._last_dialectic_turn = -999
        self._conversation_chunk_chars = 1200
        self._initialized = False

    @property
    def name(self) -> str:
        return "letta"

    def is_available(self) -> bool:
        try:
            cfg = LettaConfig.from_global_config()
            return bool(cfg.enabled and cfg.base_url)
        except Exception:
            return False

    def get_config_schema(self):
        return [
            {"key": "base_url", "description": "Letta base URL", "default": "https://letta.example.com", "required": True},
            {"key": "api_key", "description": "Letta API key (optional for open/self-hosted deployments)", "secret": True, "env_var": "LETTA_API_KEY"},
            {"key": "project_id", "description": "Optional Letta project ID", "secret": True, "env_var": "LETTA_PROJECT_ID"},
            {"key": "agent_name_prefix", "description": "Prefix for per-user Letta agents", "default": "Hermes Memory"},
            {"key": "prefetch_top_k", "description": "Default number of recall hits", "default": 5},
            {"key": "recall_mode", "description": "Recall mode for Letta memory", "choices": ["hybrid", "context", "tools"], "default": "hybrid"},
            {"key": "injection_frequency", "description": "How often auto-context is injected", "choices": ["every-turn", "first-turn"], "default": "every-turn"},
            {"key": "first_turn_context_enabled", "description": "Warm and bake Letta context on session start", "choices": [True, False], "default": True},
            {"key": "context_char_limit", "description": "Max chars of auto-injected Letta context", "default": 1200},
            {"key": "context_tokens", "description": "Approx token budget for auto-injected Letta context (0 disables token cap)", "default": 0},
            {"key": "context_cadence", "description": "Minimum turns between Letta context refreshes", "default": 1},
            {"key": "dialectic_cadence", "description": "Minimum turns between Letta query-triggered prefetches", "default": 1},
            {"key": "auto_session_summary", "description": "Store compact session summaries at session end", "choices": [True, False], "default": True},
            {"key": "session_summary_char_limit", "description": "Max chars for generated session summary text", "default": 1200},
        ]

    def save_config(self, values, hermes_home):
        config_path = Path(hermes_home) / "letta.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                existing = {}
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def test_connection(self, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cfg = LettaConfig(
                base_url=str(values.get("base_url", "")).rstrip("/"),
                api_key=str(values.get("api_key", "") or ""),
                project_id=str(values.get("project_id", "") or ""),
                agent_name_prefix=str(values.get("agent_name_prefix", "Hermes Memory") or "Hermes Memory"),
                recall_mode=_normalize_recall_mode(values.get("recall_mode") or "hybrid"),
                injection_frequency=_normalize_injection_frequency(values.get("injection_frequency") or "every-turn"),
                first_turn_context_enabled=_to_bool(values.get("first_turn_context_enabled", True), default=True),
                context_char_limit=int(values.get("context_char_limit", 1200) or 1200),
                context_tokens=int(values.get("context_tokens", 0) or 0),
                context_cadence=max(1, int(values.get("context_cadence", 1) or 1)),
                dialectic_cadence=max(1, int(values.get("dialectic_cadence", 1) or 1)),
            )
            client = LettaClient(cfg)
            client.list_agents(name="Hermes Doctor Probe")
            return {"ok": True, "detail": "reachable"}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config

        print("\n  Configuring Letta memory:\n")
        provider_config = {
            "base_url": _prompt_text("Letta base URL", default="https://letta.example.com"),
            "agent_name_prefix": _prompt_text("Agent name prefix", default="Hermes Memory"),
            "project_id": _prompt_text("Project ID (optional)", default=""),
            "recall_mode": _prompt_text("Recall mode (hybrid/context/tools)", default="hybrid"),
            "injection_frequency": _prompt_text("Injection frequency (every-turn/first-turn)", default="every-turn"),
            "first_turn_context_enabled": _prompt_yes_no("Warm first-turn context", default=True),
            "context_char_limit": int(_prompt_text("Context char limit", default="1200") or "1200"),
            "context_tokens": int(_prompt_text("Context token budget (0 disables token cap)", default="0") or "0"),
            "context_cadence": int(_prompt_text("Context cadence (minimum turns between refreshes)", default="1") or "1"),
            "dialectic_cadence": int(_prompt_text("Query cadence (minimum turns between prefetches)", default="1") or "1"),
        }
        api_key = _prompt_text("Letta API key (optional)", default="", secret=True)
        env_writes: Dict[str, str] = {}
        if api_key:
            env_writes["LETTA_API_KEY"] = api_key

        result = self.test_connection({**provider_config, "api_key": api_key})
        if not result.get("ok"):
            print(f"\n  ✗ Letta connection failed: {result.get('detail', 'unknown error')}\n")
            return

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = "letta"
        save_config(config)
        self.save_config(provider_config, hermes_home)
        if env_writes:
            _write_env_vars(Path(hermes_home) / ".env", env_writes)

        self._config = LettaConfig(
            base_url=provider_config["base_url"],
            api_key=api_key,
            project_id=provider_config["project_id"],
            agent_name_prefix=provider_config["agent_name_prefix"],
            recall_mode=provider_config["recall_mode"],
            injection_frequency=provider_config["injection_frequency"],
            first_turn_context_enabled=provider_config["first_turn_context_enabled"],
            context_char_limit=provider_config["context_char_limit"],
            context_tokens=provider_config["context_tokens"],
            context_cadence=max(1, int(provider_config.get("context_cadence", 1) or 1)),
            dialectic_cadence=max(1, int(provider_config.get("dialectic_cadence", 1) or 1)),
        )
        self._client = LettaClient(self._config)
        self._canonical_user_id = "setup:migration"
        self._agent_id = "setup-agent"
        self._initialized = True

        if _prompt_yes_no("Import existing built-in MEMORY.md and USER.md into Letta now", default=False):
            migration = self.migrate_builtin_memory(hermes_home)
            print(f"  ✓ Migration done: imported={migration.get('imported', 0)} skipped={migration.get('skipped', 0)}")

        print("\n  ✓ Letta memory configured and validated\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = LettaConfig.from_global_config()
        if not self._config.base_url:
            return
        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        if self._agent_context in ("cron", "flush", "subagent"):
            logger.debug("Letta skipped: non-primary context (%s)", self._agent_context)
            return
        self._client = LettaClient(self._config)
        self._canonical_user_id = self._resolve_canonical_user_id(kwargs)
        self._agent_id = self._ensure_agent()
        self._initialized = bool(self._agent_id)
        if self._initialized:
            self._write_profile_block()
            if self._config.first_turn_context_enabled and self._config.recall_mode in ("context", "hybrid"):
                self._refresh_prefetch("session start")
                with self._prefetch_lock:
                    self._first_turn_context = self._prefetch_result or ""

    def _resolve_canonical_user_id(self, kwargs: Dict[str, Any]) -> str:
        raw_user_id = (kwargs.get("user_id") or "").strip()
        if raw_user_id:
            if ":" in raw_user_id:
                return raw_user_id
            return f"{self._platform}:{raw_user_id}"
        return f"{self._platform}:default"

    def _safe_slug(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
        return slug[:60] or "default"

    def _agent_name(self) -> str:
        assert self._config is not None
        return f"{self._config.agent_name_prefix} {self._safe_slug(self._canonical_user_id)}"

    def _ensure_agent(self) -> str:
        assert self._client is not None
        name = self._agent_name()
        agents = self._client.list_agents(name=name)
        for agent in agents:
            if agent.get("name") == name and agent.get("id"):
                return agent["id"]
        created = self._client.create_agent(name=name, tags=["hermes", "memory", f"user:{self._safe_slug(self._canonical_user_id)}"])
        return created.get("id", "")

    def _write_profile_block(self) -> None:
        if not self._agent_id or not self._client:
            return
        profile_lines = [
            f"canonical_user_id={self._canonical_user_id}",
            f"platform={self._platform}",
            "memory_model=profile|preference|episodic|summary",
            "policy=durable_only; no raw chat dump",
        ]
        try:
            blocks = self._client.get_core_memory_blocks(self._agent_id)
            preferred = None
            for block in blocks:
                label = block.get("label") or block.get("name") or ""
                if label in ("human", "persona", "profile", "memory"):
                    preferred = label
                    break
            if preferred:
                self._client.update_core_memory_block(self._agent_id, preferred, "\n".join(profile_lines))
        except Exception as e:
            logger.debug("Letta core memory block update skipped: %s", e)

    def system_prompt_block(self) -> str:
        if not self._initialized or not self._config:
            return ""
        if self._config.recall_mode == "context":
            header = (
                "# Letta Memory\n"
                "Active (context-injection mode). Relevant Letta memory is auto-injected. "
                "No Letta memory tools are available in this mode.\n"
                "Structured memory types: profile, preference, episodic, summary."
            )
        elif self._config.recall_mode == "tools":
            header = (
                "# Letta Memory\n"
                "Active (tools-only mode). Use letta_profile / letta_search / letta_context / letta_conclude for explicit memory access.\n"
                "Structured memory types: profile, preference, episodic, summary."
            )
        else:
            header = (
                "# Letta Memory\n"
                "Active (hybrid mode). Relevant Letta memory is auto-injected and Letta tools are available.\n"
                "Structured memory types: profile, preference, episodic, summary."
            )
        return header + "\nOnly durable facts/preferences/project conventions should be written."

    def _truncate_context(self, text: str) -> str:
        limit = self._config.context_char_limit if self._config else 1200
        token_budget = (self._config.context_tokens * 4) if self._config and self._config.context_tokens else 0
        if token_budget > 0:
            limit = min(limit, token_budget) if limit > 0 else token_budget
        if limit <= 0 or len(text) <= limit:
            return text
        shortened = text[:limit]
        cut = shortened.rfind(" ")
        if cut > int(limit * 0.7):
            shortened = shortened[:cut]
        return shortened.rstrip() + " …"

    def _format_memory_record(self, content: str, *, memory_type: str, source: str, confidence: float, project: str = "", session_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "memory_type": memory_type,
            "canonical_user_id": self._canonical_user_id,
            "project": project or None,
            "content": content.strip(),
            "confidence": round(float(confidence), 3),
            "source": source,
            "session_id": session_id or self._session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            payload["metadata"] = metadata
        return json.dumps(payload, ensure_ascii=False)

    def _store_memory(self, content: str, *, memory_type: str, source: str = "user_explicit", confidence: float = 0.9, project: str = "", session_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._initialized or not self._client or not self._agent_id:
            return {"status": "skipped", "reason": "provider not initialized"}
        text = self._format_memory_record(
            content,
            memory_type=memory_type,
            source=source,
            confidence=confidence,
            project=project,
            session_id=session_id,
            metadata=metadata,
        )
        created = self._client.create_passage(self._agent_id, text)
        passage_id = ""
        if isinstance(created, dict):
            passage_id = str(created.get("id") or "")
        elif isinstance(created, list) and created:
            first = created[0]
            if isinstance(first, dict):
                passage_id = str(first.get("id") or "")
        return {"status": "stored", "id": passage_id or None, "memory_type": memory_type}

    def _decode_passage(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = item.get("text") or item.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("content"):
                payload["_raw"] = item
                return payload
        except Exception:
            return {"memory_type": "unknown", "content": text, "_raw": item}
        return None

    def _search(self, query: str, *, top_k: int = 5, memory_type: str = "", project: str = "") -> List[Dict[str, Any]]:
        if not self._initialized or not self._client or not self._agent_id:
            return []

        def _filter_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            decoded = []
            for row in rows:
                payload = self._decode_passage(row)
                if not payload:
                    continue
                if memory_type and payload.get("memory_type") != memory_type:
                    continue
                if project and payload.get("project") not in (project, None, ""):
                    continue
                decoded.append(payload)
            return decoded

        rows = self._client.search_archival_memory(self._agent_id, query=query, top_k=top_k)
        decoded = _filter_rows(rows)
        if decoded:
            return decoded

        query_terms = [term for term in re.split(r"\W+", query.lower()) if len(term) >= 3]
        archival_rows = self._client.list_archival_memory(self._agent_id, limit=max(top_k * 10, 100))
        fallback_hits = []
        for row in archival_rows:
            payload = self._decode_passage(row)
            if not payload:
                continue
            if memory_type and payload.get("memory_type") != memory_type:
                continue
            if project and payload.get("project") not in (project, None, ""):
                continue
            haystack = json.dumps(payload, ensure_ascii=False).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score > 0:
                payload["_fallback_score"] = score
                fallback_hits.append(payload)

        fallback_hits.sort(key=lambda item: item.get("_fallback_score", 0), reverse=True)
        return fallback_hits[:top_k]

    def _render_hits(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "No relevant Letta memory found."
        lines = []
        for hit in hits[:8]:
            mt = hit.get("memory_type", "unknown")
            project = hit.get("project") or "-"
            confidence = hit.get("confidence")
            content = str(hit.get("content", "")).strip()
            lines.append(f"- [{mt}] (project={project}, confidence={confidence}) {content}")
        return "\n".join(lines)

    def _refresh_prefetch(self, query: str) -> None:
        hits = self._search(query, top_k=self._config.prefetch_top_k if self._config else 5)
        text = ""
        if hits:
            text = self._truncate_context("# Letta Relevant Memory\n" + self._render_hits(hits))
        with self._prefetch_lock:
            self._prefetch_result = text

    def _should_refresh_for_query(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        lowered = q.lower()
        if any(tok in lowered for tok in [
            "remember", "kayak biasa", "seperti biasa", "preference", "prefer",
            "project", "deploy", "vault", "workflow", "sebelumnya", "lagi",
            "biasanya", "konteks", "context",
        ]):
            return True
        if len(q) >= 48:
            return True
        if len(q.split()) <= 3:
            return True
        return False

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._initialized or not self._config:
            return ""
        if self._config.recall_mode == "tools":
            return ""
        if self._config.injection_frequency == "first-turn" and self._turn_count > 1:
            return ""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            if self._first_turn_context is not None:
                result = self._truncate_context(self._first_turn_context)
                self._first_turn_context = None
                return result
            result = self._truncate_context(self._prefetch_result)
            self._prefetch_result = ""
            return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not self._initialized or not self._config:
            return
        if self._config.recall_mode == "tools":
            return
        if not self._should_refresh_for_query(query):
            return
        if self._config.injection_frequency == "first-turn" and self._turn_count > 1:
            return
        if self._config.dialectic_cadence > 1:
            if (self._turn_count - self._last_dialectic_turn) < self._config.dialectic_cadence:
                return
        self._last_dialectic_turn = self._turn_count
        if self._config.context_cadence > 1:
            if (self._turn_count - self._last_context_turn) < self._config.context_cadence:
                return
        self._last_context_turn = self._turn_count

        def _run() -> None:
            try:
                self._refresh_prefetch(query)
            except Exception:
                logger.debug("Letta queue_prefetch failed", exc_info=True)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="letta-prefetch")
        self._prefetch_thread.start()

    def _normalize_turn_text(self, text: str) -> str:
        cleaned = sanitize_context(text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_transient_turn_text(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return True
        if len(lowered) < 16:
            return True
        transient_exact = {
            "ok", "oke", "sip", "siap", "thanks", "thank you", "thx", "noted",
            "mantap", "aman", "gas", "lanjut", "sama-sama",
        }
        if lowered in transient_exact:
            return True
        if len(lowered.split()) <= 3 and lowered.replace("-", " ") in transient_exact:
            return True
        return False

    def _should_store_turn(self, user_text: str, assistant_text: str) -> bool:
        if self._is_transient_turn_text(user_text) or self._is_transient_turn_text(assistant_text):
            return False
        return bool(user_text and assistant_text)

    @staticmethod
    def _chunk_message(content: str, limit: int) -> list[str]:
        if len(content) <= limit:
            return [content]

        prefix = "[continued] "
        prefix_len = len(prefix)
        chunks = []
        remaining = content
        first = True
        while remaining:
            effective = limit if first else limit - prefix_len
            if len(remaining) <= effective:
                chunks.append(remaining if first else prefix + remaining)
                break

            segment = remaining[:effective]
            cut = segment.rfind("\n\n")
            if cut < effective * 0.3:
                cut = segment.rfind(". ")
                if cut >= 0:
                    cut += 2
            if cut < effective * 0.3:
                cut = segment.rfind(" ")
            if cut < effective * 0.3:
                cut = effective

            chunk = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
            if not first:
                chunk = prefix + chunk
            chunks.append(chunk)
            first = False

        return chunks

    def _store_turn_side(
        self,
        role: str,
        text: str,
        *,
        turn_index: int,
        session_id: str = "",
        source: str = "turn_sync",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        chunks = self._chunk_message(text, self._conversation_chunk_chars)
        chunk_total = len(chunks)
        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata = {
                "role": role,
                "turn_index": turn_index,
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            self._store_memory(
                chunk,
                memory_type="episodic",
                source=source,
                confidence=0.78,
                session_id=session_id,
                metadata=metadata,
            )

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._initialized or not self._config or not self._client or not self._agent_id:
            return
        user_text = self._normalize_turn_text(user_content)
        assistant_text = self._normalize_turn_text(assistant_content)
        if not self._should_store_turn(user_text, assistant_text):
            return
        self._turn_sequence += 1
        turn_index = self._turn_sequence

        def _sync() -> None:
            try:
                self._store_turn_side("user", user_text, turn_index=turn_index, session_id=session_id)
                self._store_turn_side("assistant", assistant_text, turn_index=turn_index, session_id=session_id)
            except Exception:
                logger.debug("Letta sync_turn failed", exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="letta-sync-turn")
        self._sync_thread.start()
        self._sync_thread.join(timeout=5.0)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = turn_number

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._config and self._config.recall_mode == "context":
            return []
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "letta_profile":
                hits = self._search("user profile preferences communication style", top_k=8)
                filtered = [h for h in hits if h.get("memory_type") in ("profile", "preference")]
                if not filtered:
                    return json.dumps({"result": "No profile facts available yet."})
                return json.dumps({"result": self._render_hits(filtered), "items": filtered})
            if tool_name == "letta_search":
                query = (args.get("query", "") or "").strip()
                if not query:
                    return tool_error("Missing required parameter: query")
                hits = self._search(
                    query,
                    top_k=int(args.get("top_k", 5) or 5),
                    memory_type=args.get("memory_type", "") or "",
                    project=args.get("project", "") or "",
                )
                if not hits:
                    return json.dumps({"result": "No relevant context found.", "items": []})
                return json.dumps({"result": self._render_hits(hits), "items": hits})
            if tool_name == "letta_context":
                query = (args.get("query", "") or "").strip()
                if not query:
                    return tool_error("Missing required parameter: query")
                hits = self._search(query, top_k=6, project=args.get("project", "") or "")
                if not hits:
                    return json.dumps({"result": "No relevant context found.", "items": []})
                return json.dumps({"result": self._render_hits(hits), "items": hits})
            if tool_name == "letta_conclude":
                conclusion = (args.get("conclusion", "") or "").strip()
                if not conclusion:
                    return tool_error("Missing required parameter: conclusion")
                stored = self._store_memory(
                    conclusion,
                    memory_type=args.get("memory_type", "preference") or "preference",
                    project=args.get("project", "") or "",
                    confidence=float(args.get("confidence", 0.9) or 0.9),
                    source=args.get("source", "user_explicit") or "user_explicit",
                )
                result = {"result": f"Conclusion saved: {conclusion}"}
                if isinstance(stored, dict):
                    result.update(stored)
                return json.dumps(result)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

    def _normalize_memory_entry(self, content: str) -> str:
        cleaned = sanitize_context(content or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _entry_hash(self, content: str, memory_type: str) -> str:
        normalized = self._normalize_memory_entry(content).lower()
        payload = f"{memory_type}|{normalized}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _history_entry_key(self, session_id: str, role: str, content: str) -> str:
        normalized = self._normalize_memory_entry(content).lower()
        payload = f"{session_id}|{role}|{normalized}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _load_existing_migrated_history_keys(self) -> set[str]:
        if not self._client or not self._agent_id:
            return set()
        keys: set[str] = set()
        try:
            for row in self._client.list_archival_memory(self._agent_id, limit=10000):
                payload = self._decode_passage(row)
                if not payload or payload.get("source") != "migrated_session_history":
                    continue
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                role = str(metadata.get("role") or "")
                session_id = str(payload.get("session_id") or "")
                content = self._normalize_memory_entry(str(payload.get("content", "")))
                if not role or not session_id or not content:
                    continue
                keys.add(self._history_entry_key(session_id, role, content))
        except Exception:
            logger.debug("Letta migration history dedup preload failed", exc_info=True)
        return keys

    def _load_existing_import_hashes(self) -> set[str]:
        if not self._client or not self._agent_id:
            return set()
        hashes: set[str] = set()
        try:
            for row in self._client.list_archival_memory(self._agent_id, limit=500):
                payload = self._decode_passage(row)
                if not payload:
                    continue
                content = self._normalize_memory_entry(str(payload.get("content", "")))
                if not content:
                    continue
                hashes.add(self._entry_hash(content, str(payload.get("memory_type", "unknown"))))
        except Exception:
            logger.debug("Letta migration dedup preload failed", exc_info=True)
        return hashes

    def _read_builtin_memory_entries(self, base_dir: str, filename: str) -> List[str]:
        path = Path(base_dir) / "memories" / filename
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return []
        entries = [self._normalize_memory_entry(item) for item in raw.split(ENTRY_DELIMITER)]
        return [item for item in entries if item]

    def migrate_builtin_memory(self, base_dir: str) -> Dict[str, Any]:
        if not self._initialized or not self._config or not self._agent_id:
            return {"imported": 0, "skipped": 0, "reason": "provider not initialized"}

        existing_hashes = self._load_existing_import_hashes()
        imported = 0
        skipped = 0
        plan = [
            ("USER.md", "profile", 0.95),
            ("MEMORY.md", "preference", 0.9),
        ]

        for filename, memory_type, confidence in plan:
            seen_local: set[str] = set()
            for entry in self._read_builtin_memory_entries(base_dir, filename):
                entry_hash = self._entry_hash(entry, memory_type)
                if entry_hash in existing_hashes or entry_hash in seen_local:
                    skipped += 1
                    continue
                result = self._store_memory(
                    entry,
                    memory_type=memory_type,
                    source="migrated_builtin_memory",
                    confidence=confidence,
                )
                if result.get("status") == "stored":
                    imported += 1
                    existing_hashes.add(entry_hash)
                    seen_local.add(entry_hash)
                else:
                    skipped += 1

        return {"imported": imported, "skipped": skipped}

    def _iter_state_db_turns(
        self,
        db_path: str,
        *,
        include_sources: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        from hermes_state import SessionDB

        selected_sources = tuple(include_sources or ("telegram", "cli"))
        db = SessionDB(db_path=Path(db_path))
        sessions = db.search_sessions(limit=100000)
        scanned = 0
        skipped = 0
        selected = 0
        turns: List[Dict[str, Any]] = []

        for session in sessions:
            scanned += 1
            session_source = str(session.get("source") or "")
            if session_source not in selected_sources:
                skipped += 1
                continue

            messages = db.get_messages(session["id"])
            pending_user: Optional[Dict[str, Any]] = None
            session_turns = 0

            for msg in messages:
                role = str(msg.get("role") or "")
                content = self._normalize_turn_text(str(msg.get("content") or ""))
                if not content:
                    continue
                if role == "user":
                    if self._is_transient_turn_text(content):
                        pending_user = None
                        continue
                    pending_user = msg | {"content": content}
                    continue
                if role != "assistant" or pending_user is None:
                    continue
                if self._is_transient_turn_text(content):
                    pending_user = None
                    continue

                session_turns += 1
                turns.append(
                    {
                        "session_id": session["id"],
                        "session_source": session_source,
                        "session_user_id": session.get("user_id") or "",
                        "session_title": session.get("title") or "",
                        "user": pending_user["content"],
                        "assistant": content,
                        "turn_index": session_turns,
                        "user_timestamp": pending_user.get("timestamp"),
                        "assistant_timestamp": msg.get("timestamp"),
                    }
                )
                pending_user = None

            if session_turns > 0:
                selected += 1
            else:
                skipped += 1

        return {
            "sessions_scanned": scanned,
            "sessions_selected": selected,
            "skipped_sessions": skipped,
            "turns": turns,
        }

    def _ensure_migration_agent(self, target_canonical_user_id: str) -> bool:
        if not self._config:
            self._config = LettaConfig.from_global_config()
        if not self._config or not self._config.base_url:
            return False
        if not self._client:
            self._client = LettaClient(self._config)
        self._initialized = True
        self._canonical_user_id = target_canonical_user_id.strip() or self._canonical_user_id
        self._agent_id = self._ensure_agent()
        self._initialized = bool(self._agent_id)
        return self._initialized

    def audit_state_db_history_migration(
        self,
        db_path: str,
        *,
        target_canonical_user_id: str,
        include_sources: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        report = self._iter_state_db_turns(db_path, include_sources=include_sources)
        turns = report.pop("turns")
        expected_keys: Counter[str] = Counter()
        for turn in turns:
            session_id = str(turn["session_id"])
            for role, text in (("user", turn["user"]), ("assistant", turn["assistant"])):
                for chunk in self._chunk_message(text, self._conversation_chunk_chars):
                    expected_keys[self._history_entry_key(session_id, role, chunk)] += 1

        if not self._ensure_migration_agent(target_canonical_user_id):
            return {
                **report,
                "expected_turns": len(turns),
                "expected_chunk_rows": sum(expected_keys.values()),
                "expected_unique_keys": len(expected_keys),
                "current_migrated_rows": 0,
                "current_unique_keys": 0,
                "duplicate_existing_rows": 0,
                "missing_expected_keys": len(expected_keys),
                "extra_unexpected_keys": 0,
                "target_canonical_user_id": target_canonical_user_id,
                "reason": "provider not initialized",
            }

        current_keys: Counter[str] = Counter()
        for row in self._client.list_archival_memory(self._agent_id, limit=10000):
            payload = self._decode_passage(row)
            if not payload or payload.get("source") != "migrated_session_history":
                continue
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            role = str(metadata.get("role") or "")
            session_id = str(payload.get("session_id") or "")
            content = str(payload.get("content") or "")
            if not role or not session_id or not content:
                continue
            current_keys[self._history_entry_key(session_id, role, content)] += 1

        return {
            **report,
            "expected_turns": len(turns),
            "expected_chunk_rows": sum(expected_keys.values()),
            "expected_unique_keys": len(expected_keys),
            "current_migrated_rows": sum(current_keys.values()),
            "current_unique_keys": len(current_keys),
            "duplicate_existing_rows": sum(count - 1 for count in current_keys.values() if count > 1),
            "missing_expected_keys": sum(1 for key in expected_keys if key not in current_keys),
            "extra_unexpected_keys": sum(1 for key in current_keys if key not in expected_keys),
            "target_canonical_user_id": self._canonical_user_id,
        }

    def cleanup_migrated_history_duplicates(
        self,
        *,
        target_canonical_user_id: str,
    ) -> Dict[str, Any]:
        if not self._ensure_migration_agent(target_canonical_user_id):
            return {
                "deleted_rows": 0,
                "duplicate_groups": 0,
                "target_canonical_user_id": target_canonical_user_id,
                "reason": "provider not initialized",
            }

        duplicate_groups = 0
        deleted_rows = 0
        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in self._client.list_archival_memory(self._agent_id, limit=10000):
            payload = self._decode_passage(row)
            if not payload or payload.get("source") != "migrated_session_history":
                continue
            row_id = str(row.get("id") or "")
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            role = str(metadata.get("role") or "")
            session_id = str(payload.get("session_id") or "")
            content = str(payload.get("content") or "")
            if not row_id or not role or not session_id or not content:
                continue
            key = self._history_entry_key(session_id, role, content)
            grouped_rows.setdefault(key, []).append(
                {
                    "row_id": row_id,
                    "created_at": str(payload.get("created_at") or ""),
                    "chunk_index": int(metadata.get("chunk_index") or 0),
                }
            )

        for items in grouped_rows.values():
            if len(items) <= 1:
                continue
            duplicate_groups += 1
            items.sort(key=lambda item: (item["created_at"], item["chunk_index"], item["row_id"]))
            for item in items[1:]:
                self._client.delete_archival_memory(self._agent_id, item["row_id"])
                deleted_rows += 1

        return {
            "deleted_rows": deleted_rows,
            "duplicate_groups": duplicate_groups,
            "target_canonical_user_id": self._canonical_user_id,
        }

    def migrate_state_db_history(
        self,
        db_path: str,
        *,
        target_canonical_user_id: str,
        include_sources: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        report = self._iter_state_db_turns(db_path, include_sources=include_sources)
        turns = report.pop("turns")
        if dry_run:
            return {
                "dry_run": True,
                **report,
                "estimated_sessions": report["sessions_selected"],
                "estimated_turns": len(turns),
                "estimated_passages": len(turns) * 2,
                "target_canonical_user_id": target_canonical_user_id,
            }

        if not self._config:
            self._config = LettaConfig.from_global_config()
        if not self._config or not self._config.base_url:
            return {
                "dry_run": False,
                **report,
                "imported_sessions": 0,
                "imported_turns": 0,
                "stored_passages": 0,
                "target_canonical_user_id": target_canonical_user_id,
                "reason": "provider not initialized",
            }

        if not self._client:
            self._client = LettaClient(self._config)

        if not self._ensure_migration_agent(target_canonical_user_id):
            return {
                "dry_run": False,
                **report,
                "imported_sessions": 0,
                "imported_turns": 0,
                "stored_passages": 0,
                "target_canonical_user_id": target_canonical_user_id,
                "reason": "target Letta agent unavailable",
            }

        stored_passages = 0
        imported_turns = 0
        imported_sessions: set[str] = set()
        existing_history_keys = self._load_existing_migrated_history_keys()

        for turn in turns:
            session_id = str(turn["session_id"])
            turn_index = int(turn["turn_index"])
            shared_metadata = {
                "session_source": turn["session_source"],
                "session_user_id": turn["session_user_id"],
                "session_title": turn["session_title"],
            }
            pending_chunks: list[tuple[str, str, Dict[str, Any]]] = []

            for role, text, original_timestamp in (
                ("user", turn["user"], turn["user_timestamp"]),
                ("assistant", turn["assistant"], turn["assistant_timestamp"]),
            ):
                chunks = self._chunk_message(text, self._conversation_chunk_chars)
                chunk_total = len(chunks)
                for chunk_index, chunk in enumerate(chunks, start=1):
                    chunk_key = self._history_entry_key(session_id, role, chunk)
                    if chunk_key in existing_history_keys:
                        continue
                    metadata = {
                        "role": role,
                        "turn_index": turn_index,
                        "chunk_index": chunk_index,
                        "chunk_total": chunk_total,
                        **shared_metadata,
                        "original_timestamp": original_timestamp,
                    }
                    pending_chunks.append((role, chunk, metadata))

            if not pending_chunks:
                continue

            for role, chunk, metadata in pending_chunks:
                self._store_memory(
                    chunk,
                    memory_type="episodic",
                    source="migrated_session_history",
                    confidence=0.78,
                    session_id=session_id,
                    metadata=metadata,
                )
                stored_passages += 1
                existing_history_keys.add(self._history_entry_key(session_id, role, chunk))

            imported_turns += 1
            imported_sessions.add(session_id)

        return {
            "dry_run": False,
            **report,
            "imported_sessions": len(imported_sessions),
            "imported_turns": imported_turns,
            "stored_passages": stored_passages,
            "target_canonical_user_id": self._canonical_user_id,
        }

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action == "remove" or not content.strip():
            return
        memory_type = "profile" if target == "user" else "preference"
        confidence = 0.95 if target == "user" else 0.9
        try:
            self._store_memory(content, memory_type=memory_type, source="builtin_memory_mirror", confidence=confidence)
        except Exception:
            logger.debug("Letta on_memory_write failed", exc_info=True)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        if not self._config or not self._config.auto_session_summary or not messages:
            return
        user_msgs = []
        for msg in messages:
            if msg.get("role") == "user":
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    user_msgs.append(text.strip())
        if not user_msgs:
            return
        summary = " | ".join(user_msgs[-5:])
        summary = summary[: self._config.session_summary_char_limit].strip()
        if summary:
            try:
                self._store_memory(summary, memory_type="summary", source="session_end_summary", confidence=0.7)
            except Exception:
                logger.debug("Letta session summary write failed", exc_info=True)

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        return


def register(ctx) -> None:
    ctx.register_memory_provider(LettaMemoryProvider())
