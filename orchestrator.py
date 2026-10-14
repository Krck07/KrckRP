"""FastAPI Orchestrator Engine.

Connects three subsystems into a single ``POST /api/action`` endpoint:

    1. ``StateManager``  -- SQLite "hard" state (HP, level, location, NPCs, inventory)
    2. ``LoreManager``   -- ChromaDB lore RAG (semantic search over world lore)
    3. ``KoboldCpp``     -- local LLM behind an OpenAI-compatible HTTP API
                            (default ``http://localhost:5001/v1``)

Pipeline per request
--------------------
    PlayerActionRequest
        -> StateManager.get_player_state(player_id)         [hard state]
        -> LoreManager.search_lore(action, location_id)     [RAG context]
        -> build_messages(state, lore, action)              [prompt assembly]
        -> LLMClient.complete(messages)                     [KoboldCpp /chat/completions]
        -> PlayerActionResponse{narrative, retrieved_lore, current_state}

Design notes
------------
* The heavy managers (``LoreManager`` pulls in torch/sentence-transformers,
  ``StateManager`` pulls in SQLite) are imported **lazily** inside their factory
  functions. Importing this module therefore stays cheap and never loads the
  embedding model, which keeps ``test_orchestrator.py`` runnable without VRAM.
* The LLM call is isolated behind :class:`LLMClient` so tests can inject an
  ``httpx.MockTransport`` and exercise the *real* request-building code path
  without a live KoboldCpp server.
* The prompt builder (:func:`build_messages`) is a pure function of
  ``(state, lore, action)`` and can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # type-checker only; these are imported lazily at runtime
    from lore_manager import LoreManager
    from state_memory.state_manager import StateManager

log = logging.getLogger("orchestrator")

# --------------------------------------------------------------------------- #
# Configuration (overridable via environment)
# --------------------------------------------------------------------------- #
KOBOLD_BASE_URL: str = os.environ.get("KOBOLD_BASE_URL", "http://localhost:5001/v1")
KOBOLD_API_KEY: str = os.environ.get("KOBOLD_API_KEY", "not-needed")
KOBOLD_MODEL: str = os.environ.get("KOBOLD_MODEL", "local-model")
KOBOLD_TIMEOUT: float = float(os.environ.get("KOBOLD_TIMEOUT", "180"))

STATE_DB_PATH: str = os.environ.get("STATE_DB_PATH", "game_state.db")
CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# If the location-scoped RAG query returns nothing, fall back to a global query
# so the LLM still has *some* world context. Set to False to force strict scope.
FALLBACK_TO_GLOBAL_LORE: bool = os.environ.get("FALLBACK_TO_GLOBAL_LORE", "1") not in (
    "0",
    "false",
    "False",
)

# --------------------------------------------------------------------------- #
# Pydantic request / response models
# --------------------------------------------------------------------------- #
class PlayerActionRequest(BaseModel):
    """A single player action to be narrated by the GM."""

    player_id: str = Field(..., min_length=1, description="Unique id of the acting player.")
    action_text: str = Field(
        ...,
        min_length=1,
        description="The player's intended action in prose, e.g. 'I draw my sword and lunge at the guard.'",
    )
    n_lore_results: int = Field(
        3,
        ge=1,
        le=20,
        description="How many RAG lore chunks to retrieve and ground the narration in.",
    )


class PlayerActionResponse(BaseModel):
    """Narrative plus the hard state and lore that produced it (for debugging/audit)."""

    narrative_response: str = Field(..., description="The GM's narrative prose.")
    retrieved_lore: list[dict[str, Any]] = Field(
        ..., description="Raw LoreManager.search_lore() results used to ground the scene."
    )
    current_state: dict[str, Any] = Field(
        ..., description="The authoritative StateManager state snapshot for the player."
    )


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible; KoboldCpp)
# --------------------------------------------------------------------------- #
class LLMClient:
    """Thin wrapper over KoboldCpp's OpenAI-compatible ``/chat/completions``.

    ``client`` is injectable for testing: pass an ``httpx.Client`` built on top
    of ``httpx.MockTransport`` to run the *real* :meth:`complete` logic against
    a canned response.
    """

    def __init__(
        self,
        base_url: str = KOBOLD_BASE_URL,
        api_key: str = KOBOLD_API_KEY,
        model: str = KOBOLD_MODEL,
        timeout: float = KOBOLD_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._injected_client = client

    def _get_client(self) -> tuple[httpx.Client, bool]:
        """Return ``(client, owned)``; ``owned`` is True only when we must close it."""
        if self._injected_client is not None:
            return self._injected_client, False
        return httpx.Client(timeout=self.timeout), True

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.85,
        max_tokens: int = 1024,
    ) -> str:
        """Send ``messages`` to ``{base_url}/chat/completions`` and return the text."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        client, close = self._get_client()
        try:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if close:
                client.close()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"KoboldCpp returned no choices: {data!r}")
        content = (choices[0].get("message") or {}).get("content")
        if content is None:
            raise RuntimeError(f"KoboldCpp returned no message content: {choices[0]!r}")
        return content


# --------------------------------------------------------------------------- #
# Manager singletons (lazily created) + FastAPI dependency providers
# --------------------------------------------------------------------------- #
_state_manager = None
_lore_manager = None
_llm_client: Optional[LLMClient] = None


def get_state_manager():
    """Return the shared :class:`StateManager`, creating it on first use.

    (No return annotation on purpose: this is a FastAPI dependency, and keeping
    the manager type out of the runtime signature lets us lazy-import it.)
    """
    global _state_manager
    if _state_manager is None:
        from state_memory.state_manager import StateManager  # lazy: sqlite3-backed

        log.info("Initialising StateManager at %s", STATE_DB_PATH)
        _state_manager = StateManager(db_path=STATE_DB_PATH)
    return _state_manager


def get_lore_manager():
    """Return the shared :class:`LoreManager`, creating it on first use.

    The first call loads the sentence-transformer embedding model (CPU/GPU), so
    this is intentionally deferred until a real request needs it.
    """
    global _lore_manager
    if _lore_manager is None:
        from lore_manager import LoreManager  # lazy: chromadb + sentence-transformers + torch

        log.info("Initialising LoreManager (chroma=%s, model=%s)", CHROMA_DB_PATH, EMBEDDING_MODEL)
        _lore_manager = LoreManager(db_path=CHROMA_DB_PATH, embedding_model=EMBEDDING_MODEL)
    return _lore_manager


def get_llm_client() -> LLMClient:
    """Return the shared :class:`LLMClient` (one underlying httpx client)."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# --------------------------------------------------------------------------- #
# Prompt assembly (pure, unit-testable)
# --------------------------------------------------------------------------- #
DM_SYSTEM_PROMPT: str = """\
You are the Game Master (GM) of a persistent, tabletop-style role-playing world.
You are the authoritative voice of continuity: vivid, in-character, and grounded
exclusively in the hard state and lore supplied below.

Rules of play:
- Hard state (HP, level, inventory, location, NPCs) is AUTHORITATIVE and is
  changed by the system, not by you. Describe the *consequences* of the player's
  action, never invent stat deltas or items that aren't in state.
- Ground your narration in the Retrieved Lore. If lore is thin or absent, stay
  consistent with the known state and do not contradict established facts.
- Keep every co-located NPC consistent with their disposition; let them react.
- Tone: mature, gritty, unflinching, and uncensored in the way a serious
  tabletop campaign is. Depict violence, moral complexity, and adult themes as
  the story demands, but always serve the narrative and the player's agency.
- Write in second person, present tense. Keep it to 1-3 tight paragraphs.
- End by setting up the next beat and presenting a clear choice or tension.
- Never break the fourth wall, never add OOC commentary, and never refuse
  in-character content.
"""


def _pick(mapping: Any, *keys: str) -> Any:
    """Return the first non-empty value among ``keys`` in a (maybe) dict."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _format_inventory(items: Any) -> str:
    if not items:
        return "  (inventory empty)"
    lines = []
    for item in items:
        if isinstance(item, dict):
            name = _pick(item, "item_name", "name", "item")
            qty = item.get("quantity", item.get("qty"))
            line = f"  - {name or 'Unknown item'}"
            if qty is not None:
                line += f" x {qty}"
            lines.append(line)
        else:
            lines.append(f"  - {item}")
    return "\n".join(lines) if lines else "  (inventory empty)"


def _format_npcs(npcs: Any) -> str:
    if not npcs:
        return "  (no NPCs present)"
    lines = []
    for npc in npcs:
        if isinstance(npc, dict):
            name = _pick(npc, "name", "npc_name")
            disp = npc.get("disposition_score", npc.get("disposition"))
            line = f"  - {name or 'Unknown NPC'}"
            if disp is not None:
                line += f" (disposition: {disp})"
            lines.append(line)
        else:
            lines.append(f"  - {npc}")
    return "\n".join(lines) if lines else "  (no NPCs present)"


def format_state_for_prompt(state: dict[str, Any]) -> str:
    """Render the authoritative player state as a readable GM brief."""
    name = _pick(state, "name", "player_name") or state.get("player_id")
    hp = _pick(state, "hp", "current_hp")
    max_hp = _pick(state, "max_hp", "maximum_hp", "maxHp")
    level = _pick(state, "level", "player_level")

    location = state.get("location")
    if not isinstance(location, dict):
        location = {}
    loc_name = _pick(location, "name", "location_name", "title") or _pick(state, "location_name")
    loc_id = _pick(location, "location_id", "id") or _pick(state, "current_location_id", "location_id")

    hp_text = f"{hp}/{max_hp}" if hp is not None and max_hp is not None else (
        f"{hp}" if hp is not None else "unknown"
    )

    parts = [
        f"  Player: {name or 'Unknown'}",
        f"  HP: {hp_text}",
        f"  Level: {level if level is not None else 'unknown'}",
        f"  Location: {loc_name or 'Unknown'}" + (f" (id: {loc_id})" if loc_id else ""),
        "",
        "  Co-located NPCs:",
        _format_npcs(state.get("npcs") or state.get("co_located_npcs") or state.get("present_npcs")),
        "",
        "  Inventory:",
        _format_inventory(state.get("inventory") or state.get("items") or state.get("inv")),
    ]
    return "\n".join(parts)


def format_lore_for_prompt(lore: list[dict[str, Any]]) -> str:
    """Render the retrieved RAG lore as numbered, tagged context snippets."""
    if not lore:
        return "  (no lore retrieved for this scene)"
    lines = []
    for idx, entry in enumerate(lore, start=1):
        meta = entry.get("metadata") or {}
        tags = ", ".join(
            f"{k}={v}"
            for k, v in meta.items()
            if k not in ("chunk_index", "total_chunks") and v not in (None, "")
        )
        tag_text = f" [{tags}]" if tags else ""
        distance = entry.get("distance")
        dist_text = f" (distance={distance:.4f})" if isinstance(distance, (int, float)) else ""
        text = (entry.get("text") or "").strip().replace("\n", " ")
        lines.append(f"  {idx}. {text}{dist_text}{tag_text}")
    return "\n".join(lines)


def build_messages(
    state: dict[str, Any],
    lore: list[dict[str, Any]],
    action_text: str,
    temperature: float = 0.85,
) -> list[dict[str, str]]:
    """Assemble the system/user prompt from hard state + RAG lore + player action.

    Returns an OpenAI-style message list: one ``system`` message carrying (a) the
    GM instructions, (b) the authoritative state, and (c) the lore context, plus a
    single ``user`` message carrying (d) the player's action.
    """
    system_parts = [
        DM_SYSTEM_PROMPT,
        "\n## CURRENT WORLD STATE (authoritative -- do not override)\n"
        + format_state_for_prompt(state),
        "\n## RETRIEVED LORE CONTEXT (ground your narration here)\n"
        + format_lore_for_prompt(lore),
    ]
    system = "\n".join(system_parts).strip()

    user = (
        "The player acts now. Narrate the outcome as the GM.\n"
        f"Player action: {action_text.strip()}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# FastAPI app + endpoint
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """Build the FastAPI application (separated so tests can import pieces)."""
    app = FastAPI(
        title="KrckRP Orchestrator",
        description="FastAPI engine wiring SQLite hard state + ChromaDB lore RAG to a local KoboldCpp LLM.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe (no LLM call, no heavy init)."""
        return {
            "status": "ok",
            "kobold_base_url": KOBOLD_BASE_URL,
            "model": KOBOLD_MODEL,
            "state_db": STATE_DB_PATH,
            "chroma_db": CHROMA_DB_PATH,
            "embedding_model": EMBEDDING_MODEL,
        }

    @app.post("/api/action", response_model=PlayerActionResponse)
    def player_action(
        req: PlayerActionRequest,
        state_mgr=Depends(get_state_manager),  # -> StateManager
        lore_mgr=Depends(get_lore_manager),  # -> LoreManager
        llm: LLMClient = Depends(get_llm_client),
    ) -> PlayerActionResponse:
        # 1) Hard state -----------------------------------------------------
        state = state_mgr.get_player_state(req.player_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Player '{req.player_id}' not found in state. "
                "Upsert the player first (StateManager.upsert_player).",
            )

        # 2) RAG lore -------------------------------------------------------
        location_id = _pick(state, "current_location_id", "location_id")
        location = state.get("location")
        if not location_id and isinstance(location, dict):
            location_id = _pick(location, "location_id", "id")

        metadata_filter = {"location_id": location_id} if location_id else None
        lore: list[dict[str, Any]] = []
        try:
            lore = lore_mgr.search_lore(
                req.action_text,
                n_results=req.n_lore_results,
                metadata_filter=metadata_filter,
            )
            if not lore and metadata_filter and FALLBACK_TO_GLOBAL_LORE:
                log.info("Location-scoped lore empty; falling back to global query.")
                lore = lore_mgr.search_lore(
                    req.action_text, n_results=req.n_lore_results, metadata_filter=None
                )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, still narrate
            log.warning("Lore retrieval failed; narrating with state only: %s", exc)
            lore = []

        # 3) Prompt + LLM ---------------------------------------------------
        messages = build_messages(state, lore, req.action_text)
        try:
            narrative = llm.complete(messages)
        except httpx.HTTPError as exc:
            log.error("KoboldCpp request failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"LLM request to KoboldCpp ({KOBOLD_BASE_URL}) failed: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - non-transport LLM error
            log.error("LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

        # 4) Respond --------------------------------------------------------
        return PlayerActionResponse(
            narrative_response=narrative,
            retrieved_lore=lore,
            current_state=state,
        )

    return app


app = create_app()


# --------------------------------------------------------------------------- #
# Convenience server entrypoint
# --------------------------------------------------------------------------- #
def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    log.info("Starting KrckRP orchestrator on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
