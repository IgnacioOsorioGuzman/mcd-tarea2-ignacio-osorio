from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from minichat.core.events import Event
from typing import Any, Optional

from minichat.core.backend import Backend
from minichat.core.observable import Observable
from minichat.model.chat_session import ChatSession
from minichat.model.generation_config import GenerationConfig
from minichat.model.role import Role

logger = logging.getLogger(__name__)


class ChatController(Observable):
    """Coordinates the conversation between session and backend.

    Emits the following observable events:
        - CONFIG_CHANGED: after updating GenerationConfig
        - GENERATING_START: before calling the backend
        - GENERATING_DONE: after a reply (payload: {"reply": str})
        - GENERATING_ERROR: on backend exception
          (payload: {"error": str})
        - MESSAGES_UPDATED: after session mutations
        - EXPORT_DONE: after exporting JSON (payload: {"path": str})
    """

    def __init__(
        self,
        session: ChatSession,
        backend: Backend,
        *,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize with dependencies and default config.

        Note:
            Dependencies are injected to ease testing with fakes/mocks.
        """
        super().__init__()
        self._session = session
        self._backend = backend
        self._cfg = GenerationConfig()
        self._log = logger_ or logger

    @property
    def session(self) -> ChatSession:
        """Read-only access to the session."""
        return self._session

    @property
    def backend(self) -> Backend:
        """Read-only access to the backend."""
        return self._backend

    @property
    def config(self) -> GenerationConfig:
        """Current generation configuration."""
        return self._cfg

    def set_generation(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        max_ctx_tokens: int,
    ) -> None:
        """Update generation parameters.

        Raises:
            ValueError: If any value is invalid.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_ctx_tokens <= 0:
            raise ValueError("max_ctx_tokens must be positive")

        self._cfg = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            max_ctx_tokens=max_ctx_tokens,
        )
        self._notify(Event.CONFIG_CHANGED, {"cfg": asdict(self._cfg)})

    def send_user(self, text: str) -> None:
        """Add a user message and generate an assistant reply."""

        text = self._validate_text(text)
        if not text: 
            return

        self._session.add(Role.USER, text)
        self._notify(Event.GENERATING_START, {})

        try:
            reply, elapsed_ms = self._generate_reply()
        except Exception as exc:
            self._notify(Event.GENERATING_ERROR, {"error": str(exc)}) 
            return

        meta = self._build_meta(text, reply, elapsed_ms)
        self._session.add(Role.ASSISTANT, reply, meta)
        self._notify(Event.GENERATING_DONE, {"reply": reply})

    def add_assistant(self, text: str) -> None:
        """Manually append an assistant message (no generation)."""
        text = self._validate_text(text)
        if not text:
            return
        self._session.add(Role.ASSISTANT, text)
        self._notify(Event.MESSAGES_UPDATED, {})

    def add_system(self, text: str) -> None:
        """Append a system message."""
        text = self._validate_text(text)
        if not text:
            return
        self._session.add(Role.SYSTEM, text)
        self._notify(Event.MESSAGES_UPDATED, {})

    def delete_at(self, index: int) -> None:
        """Delete the message at the given index."""
        self._session.delete_at(index)
        self._notify(Event.MESSAGES_UPDATED, {})

    def clear_all(self) -> None:
        """Remove every message from the session."""
        self._session.clear()
        self._notify(Event.MESSAGES_UPDATED, {})

    def export_json(self, path: str = "chat_transcript.json") -> None:
        """Export the conversation transcript as JSON."""
        data = self._session.as_public()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        self._notify(Event.EXPORT_DONE, {"path": path})

    def _generate_reply(self) -> tuple[str, int]:
        """Generate a reply and return (reply, latency_ms)."""

        started = time.perf_counter()

        reply = self._backend.reply(
            self._session,
            self._cfg,
        )

        elapsed_ms = int(
            (time.perf_counter() - started) * 1000
        )

        return reply, elapsed_ms

    def _build_meta(
        self,
        text: str,
        reply: str,
        elapsed_ms: int,
        ) -> dict:
        """Build assistant metadata."""

        return {
            "backend": getattr(self._backend, "name", "unknown"),
            "gen_cfg": asdict(self._cfg),
            "latency_ms": elapsed_ms,
            "prompt_chars": len(text),
            "reply_chars": len(reply),
        }


    def _validate_text(self, text: Optional[str]) -> str:
        """Return normalized text."""
        text = (text or "").strip()

        if len(text) > 2000:
            text = text[:2000]

        return text
