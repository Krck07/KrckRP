"""Tests for ``orchestrator.py``.

Two modes
---------
Default (no args) -- fast, offline mock tests. These NEVER touch KoboldCpp and
NEVER load the embedding model:

    * ``build_messages()`` is tested as a pure function (prompt construction).
    * The full ``POST /api/action`` pipeline is exercised through FastAPI's
      ``TestClient`` with the state + lore managers replaced by fakes and the
      LLM replaced by an ``httpx.MockTransport``. Because we mock at the
      httpx transport boundary, the *real* ``LLMClient.complete()`` code path
      (URL, auth header, payload, response parsing) runs against a canned
      OpenAI-compatible response.

Live (``--live``) -- a real integration run. Requires KoboldCpp running on port
5001 and free GPU/VRAM (it loads the sentence-transformer embedding model and
actually calls the LLM). Run manually, later:

    python test_orchestrator.py --live
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
from fastapi.testclient import TestClient

# Ensure the repo root is importable regardless of where we run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import (  # noqa: E402
    KOBOLD_BASE_URL,
    KOBOLD_MODEL,
    LLMClient,
    build_messages,
    app,
    get_llm_client,
    get_lore_manager,
    get_state_manager,
)

CANNED_NARRATIVE = (
    "You wade chest-deep into the murk, your blade flashing as the warden turns. "
    "The iron bells of the Brotherhood ring out across the black water, and the "
    "swamp itself seems to hold its breath. Two more wardens rise from the reeds "
    "-- do you press the attack, or break off and slip away?"
)

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _sample_state(player_id: str) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "name": "Kaelen",
        "hp": 42,
        "max_hp": 60,
        "level": 5,
        "current_location_id": "oakhaven_swamp",
        "location": {
            "location_id": "oakhaven_swamp",
            "name": "Oakhaven Swamp",
            "base_description": "A fog-choked marsh of black water and dead cypress.",
        },
        "npcs": [{"name": "Vaelen", "disposition_score": -3}],
        "inventory": [
            {"item_id": "torch_1", "owner_id": player_id, "item_name": "Torch", "quantity": 1},
            {"item_id": "potion_1", "owner_id": player_id, "item_name": "Healing Potion", "quantity": 2},
        ],
    }


def _sample_lore() -> list[dict[str, Any]]:
    return [
        {
            "id": "chunk_a",
            "text": "The Obsidian Brotherhood keeps its vigil in the black bogs, and their iron bells toll at dusk.",
            "metadata": {"lore_type": "faction", "location_id": "oakhaven_swamp"},
            "distance": 0.18,
        },
        {
            "id": "chunk_b",
            "text": "Vaelen once served the Tideborn and carries a debt he cannot repay.",
            "metadata": {"lore_type": "character", "character_id": "vaelen", "location_id": "oakhaven_swamp"},
            "distance": 0.27,
        },
    ]


class FakeState:
    """Duck-typed stand-in for StateManager (only get_player_state is used)."""

    def __init__(self, state: dict[str, Any] | None) -> None:
        self._state = state
        self.last_player: str | None = None

    def get_player_state(self, player_id: str) -> dict[str, Any] | None:
        self.last_player = player_id
        return self._state if self._state and self._state.get("player_id") == player_id else None


class FakeLore:
    """Duck-typed stand-in for LoreManager (only search_lore is used)."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    def search_lore(self, query: str, n_results: int = 3, metadata_filter: dict | None = None):
        self.calls.append({"query": query, "n_results": n_results, "metadata_filter": metadata_filter})
        return list(self._results)


def _llm_client_capturing(captured: dict[str, Any], narrative: str = CANNED_NARRATIVE) -> LLMClient:
    """Build an LLMClient on a MockTransport that records the request."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": "cmpl_mock", "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": narrative}}
            ]},
        )

    return LLMClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _llm_client_failing() -> LLMClient:
    """Build an LLMClient whose transport simulates 'connection refused'."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused (simulated)", request=request)

    return LLMClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _override(state: FakeState | None, lore: FakeLore, llm: LLMClient) -> None:
    app.dependency_overrides[get_state_manager] = lambda: state
    app.dependency_overrides[get_lore_manager] = lambda: lore
    app.dependency_overrides[get_llm_client] = lambda: llm


def _reset_overrides() -> None:
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_build_messages_contains_state_lore_and_action() -> None:
    state = _sample_state("kaelen")
    lore = _sample_lore()
    action = "I draw my blade and charge the Brotherhood warden."

    messages = build_messages(state, lore, action)

    assert [m["role"] for m in messages] == ["system", "user"]
    system = messages[0]["content"]
    user = messages[1]["content"]

    # (a) DM instructions
    assert "Game Master" in system
    # (b) hard state: hp/max_hp, level, location, inventory, npc
    assert "42/60" in system
    assert "Level: 5" in system
    assert "Oakhaven Swamp" in system
    assert "Torch" in system and "Healing Potion" in system
    assert "Vaelen" in system
    # (c) retrieved lore
    assert "Obsidian Brotherhood" in system
    assert "iron bells" in system
    # (d) player action lives in the user message
    assert "charge the Brotherhood warden" in user
    assert "I draw my blade" in user
    print("[PASS] build_messages contains state + lore + action")


def test_build_messages_tolerates_missing_location_and_empty_lore() -> None:
    state = {"player_id": "x", "name": "Rook", "hp": 10, "max_hp": 10, "level": 1}
    messages = build_messages(state, [], "I look around.")
    system = messages[0]["content"]
    assert "Rook" in system
    assert "10/10" in system
    assert "no lore retrieved" in system
    print("[PASS] build_messages tolerates missing location + empty lore")


def test_full_pipeline_success() -> None:
    state = _sample_state("kaelen")
    fake_state = FakeState(state)
    fake_lore = FakeLore(_sample_lore())
    captured: dict[str, Any] = {}
    fake_llm = _llm_client_capturing(captured)

    _override(fake_state, fake_lore, fake_llm)
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/action",
            json={
                "player_id": "kaelen",
                "action_text": "I draw my blade and charge the Brotherhood warden.",
                "n_lore_results": 3,
            },
        )
    finally:
        _reset_overrides()

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Response shape
    assert body["narrative_response"] == CANNED_NARRATIVE
    assert body["retrieved_lore"] == _sample_lore()
    assert body["current_state"]["hp"] == 42
    assert body["current_state"]["level"] == 5

    # The real LLMClient hit the right endpoint with the right model + messages.
    assert captured["url"].endswith("/v1/chat/completions"), captured["url"]
    assert captured["payload"]["model"] == KOBOLD_MODEL == "local-model"
    assert captured["headers"].get("authorization") == "Bearer not-needed"

    # The prompt the LLM received is grounded in state + lore + action.
    msgs = captured["payload"]["messages"]
    system = next(m for m in msgs if m["role"] == "system")["content"]
    user = next(m for m in msgs if m["role"] == "user")["content"]
    assert "42/60" in system
    assert "Oakhaven Swamp" in system
    assert "Obsidian Brotherhood" in system
    assert "charge the Brotherhood warden" in user

    # RAG used the player's current location as the metadata filter.
    assert fake_lore.calls[0]["metadata_filter"] == {"location_id": "oakhaven_swamp"}
    assert fake_lore.calls[0]["n_results"] == 3
    assert fake_state.last_player == "kaelen"
    print("[PASS] full pipeline success (state + lore + action -> narrative)")


def test_pipeline_falls_back_to_global_lore_when_scoped_empty() -> None:
    state = _sample_state("kaelen")
    fake_lore = FakeLore([])  # first (scoped) call returns nothing
    fake_state = FakeState(state)
    captured: dict[str, Any] = {}
    fake_llm = _llm_client_capturing(captured)

    # Simulate: scoped query empty -> orchestrator falls back to a global query.
    # FakeLore returns [] for BOTH calls, so the pipeline still succeeds.
    _override(fake_state, fake_lore, fake_llm)
    try:
        client = TestClient(app)
        resp = client.post("/api/action", json={"player_id": "kaelen", "action_text": "I listen."})
    finally:
        _reset_overrides()

    assert resp.status_code == 200, resp.text
    # Two calls: one scoped (with filter), one global (filter=None) fallback.
    assert len(fake_lore.calls) == 2, fake_lore.calls
    assert fake_lore.calls[0]["metadata_filter"] == {"location_id": "oakhaven_swamp"}
    assert fake_lore.calls[1]["metadata_filter"] is None
    print("[PASS] falls back to global lore when scoped query is empty")


def test_pipeline_player_not_found_returns_404() -> None:
    fake_lore = FakeLore(_sample_lore())
    fake_llm = _llm_client_capturing({})
    _override(FakeState(None), fake_lore, fake_llm)
    try:
        client = TestClient(app)
        resp = client.post("/api/action", json={"player_id": "ghost", "action_text": "I attack."})
    finally:
        _reset_overrides()

    assert resp.status_code == 404, resp.text
    assert "not found" in resp.json()["detail"].lower()
    # No lore query and no LLM call should have happened.
    assert fake_lore.calls == []
    print("[PASS] unknown player -> 404, no RAG/LLM call")


def test_pipeline_llm_failure_returns_502() -> None:
    state = _sample_state("kaelen")
    fake_llm = _llm_client_failing()
    _override(FakeState(state), FakeLore(_sample_lore()), fake_llm)
    try:
        client = TestClient(app)
        resp = client.post("/api/action", json={"player_id": "kaelen", "action_text": "I attack."})
    finally:
        _reset_overrides()

    assert resp.status_code == 502, resp.text
    assert "KoboldCpp" in resp.json()["detail"]
    print("[PASS] LLM failure -> 502 with a clear detail")


def test_validation_rejects_bad_n_lore() -> None:
    _override(FakeState(_sample_state("kaelen")), FakeLore(_sample_lore()), _llm_client_capturing({}))
    try:
        client = TestClient(app)
        resp = client.post("/api/action", json={"player_id": "kaelen", "action_text": "I act.", "n_lore_results": 0})
    finally:
        _reset_overrides()

    assert resp.status_code == 422, resp.text
    print("[PASS] n_lore_results < 1 -> 422 validation error")


def _run_mock_tests() -> int:
    tests = [
        test_build_messages_contains_state_lore_and_action,
        test_build_messages_tolerates_missing_location_and_empty_lore,
        test_full_pipeline_success,
        test_pipeline_falls_back_to_global_lore_when_scoped_empty,
        test_pipeline_player_not_found_returns_404,
        test_pipeline_llm_failure_returns_502,
        test_validation_rejects_bad_n_lore,
    ]
    failures = 0
    print("== KrckRP orchestrator -- offline mock tests ==")
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[ERROR] {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"== {len(tests) - failures}/{len(tests)} passed ==")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Live integration (opt-in, requires KoboldCpp on :5001 + free VRAM)
# --------------------------------------------------------------------------- #
def run_live_integration() -> int:
    """Real end-to-end run: real managers + real KoboldCpp.

    Uses the app's *shared* manager singletons so the embedding model loads only
    once. Seeds a small player/location/inventory/lore set, then POSTs a real
    action and prints the full response.
    """
    print("== KrckRP orchestrator -- LIVE integration (real LLM) ==")
    print(f"Kobold base URL: {KOBOLD_BASE_URL}")
    print("WARNING: this loads the embedding model and calls the LLM. "
          "Ensure KoboldCpp is running and VRAM is free.\n")

    # Seed hard state (SQLite only -- cheap).
    state_mgr = get_state_manager()
    state_mgr.upsert_location(
        location_id="oakhaven_swamp",
        name="Oakhaven Swamp",
        base_description="A fog-choked marsh of black water, dead cypress, and iron bells.",
    )
    state_mgr.upsert_npc(npc_id="vaelen", name="Vaelen", location_id="oakhaven_swamp", disposition_score=-3)
    state_mgr.upsert_player(
        player_id="kaelen", name="Kaelen", hp=42, max_hp=60, level=5,
        current_location_id="oakhaven_swamp",
    )
    state_mgr.upsert_inventory_item(item_id="torch_1", owner_id="kaelen", item_name="Torch", quantity=1)
    state_mgr.upsert_inventory_item(item_id="potion_1", owner_id="kaelen", item_name="Healing Potion", quantity=2)

    # Seed lore (this loads the sentence-transformer model -- the heavy step).
    lore_mgr = get_lore_manager()
    lore_mgr.insert_lore(
        text=(
            "The Obsidian Brotherhood keeps its vigil in the black bogs of the "
            "Oakhaven Swamp, and their iron bells toll at dusk. Vaelen once served "
            "the Tideborn and carries a debt he cannot repay. The marsh remembers "
            "every name spoken over its dead water."
        ),
        lore_type="faction",
        location_id="oakhaven_swamp",
        character_id="vaelen",
    )

    # Real request through the real app (no dependency overrides).
    client = TestClient(app)
    resp = client.post(
        "/api/action",
        json={"player_id": "kaelen", "action_text": "I draw my blade and charge the Brotherhood warden.",
              "n_lore_results": 3},
    )
    print(f"HTTP {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    return 0 if resp.status_code == 200 else 1


def main() -> int:
    if "--live" in sys.argv:
        return run_live_integration()
    return _run_mock_tests()


if __name__ == "__main__":
    raise SystemExit(main())
