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
    app,
    build_messages,
    get_llm_client,
    get_lore_manager,
    get_state_manager,
    parse_and_apply_mutations,
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
    """Fake state in the EXACT nested shape of StateManager.get_player_state():
    stats under ``player``, location under ``location``, co-located NPCs under
    ``npcs_here``, plus ``inventory`` and ``quests`` at the top level."""
    return {
        "player": {
            "id": player_id,
            "name": "Kaelen",
            "hp": 42,
            "max_hp": 60,
            "level": 5,
            "current_location_id": "oakhaven_swamp",
        },
        "location": {
            "id": "oakhaven_swamp",
            "name": "Oakhaven Swamp",
            "base_description": "A fog-choked marsh of black water and dead cypress.",
        },
        "npcs_here": [
            {"id": "vaelen", "name": "Vaelen", "disposition_score": -3},
        ],
        "inventory": [
            {"item_id": "torch_1", "item_name": "Torch", "quantity": 1},
            {"item_id": "potion_1", "item_name": "Healing Potion", "quantity": 2},
        ],
        "quests": [
            {
                "quest_id": "swamp_mystery",
                "title": "The Swamp's Mystery",
                "description": "Learn why the Brotherhood's iron bells toll for the dead.",
                "status": "active",
            },
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
    """Duck-typed stand-in for StateManager (implements every method the
    endpoint and the mutation parser call)."""

    def __init__(self, state: dict[str, Any] | None) -> None:
        self._state = state
        self.last_player: str | None = None
        self.chat_turns: list[tuple[str, str, str]] = []
        self.hp_updates: list[tuple[str, int]] = []
        self.resource_updates: list[tuple[str, int]] = []
        self.ability_adds: list[tuple[str, str, str, int]] = []
        self.inventory_adds: list[tuple[str, str, str, int]] = []
        self.inventory_removes: list[tuple[str, str]] = []

    def get_player_state(self, player_id: str) -> dict[str, Any] | None:
        self.last_player = player_id
        if self._state is None:
            return None
        # Nested shape (get_player_state()): player id lives under state["player"]["id"].
        player = self._state.get("player")
        if isinstance(player, dict) and player.get("id") == player_id:
            return self._state
        # Tolerate a flat state dict (legacy shape) as a fallback.
        if self._state.get("player_id") == player_id:
            return self._state
        return None

    def get_recent_chat(self, player_id: str, limit: int = 15) -> list[dict[str, str]]:
        return []

    def append_chat_turn(self, player_id: str, role: str, content: str) -> None:
        self.chat_turns.append((player_id, role, content))

    def update_player_hp(self, player_id: str, hp_change: int) -> None:
        self.hp_updates.append((player_id, hp_change))

    def update_player_resource(self, player_id: str, change: int) -> None:
        self.resource_updates.append((player_id, change))

    def upsert_ability(
        self,
        player_id: str,
        ability_id: str,
        name: str,
        resource_cost: int,
    ) -> None:
        self.ability_adds.append((player_id, ability_id, name, resource_cost))

    def upsert_inventory_item(
        self, item_id: str, owner_id: str, item_name: str, quantity: int
    ) -> None:
        self.inventory_adds.append((item_id, owner_id, item_name, quantity))

    def remove_inventory_item(self, item_id: str, owner_id: str) -> None:
        self.inventory_removes.append((item_id, owner_id))


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


def test_build_messages_splices_chat_history() -> None:
    state = _sample_state("kaelen")
    history = [
        {"role": "user", "content": "I search the reeds."},
        {"role": "assistant", "content": "You find a rusted key."},
        {"role": "system", "content": "should be dropped"},   # invalid role
        {"role": "user", "content": "   "},                  # blank content
    ]
    messages = build_messages(state, _sample_lore(), "I head north.", chat_history=history)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"], roles
    assert messages[1]["content"] == "I search the reeds."
    assert messages[2]["content"] == "You find a rusted key."
    assert "I head north." in messages[3]["content"]
    print("[PASS] build_messages splices valid chat history between system and action")


def test_parse_and_apply_mutations_applies_and_cleans() -> None:
    fake = FakeState(_sample_state("kaelen"))
    narrative = (
        "The warden's bell cracks across your skull. You stumble, bleeding into the murk.\n\n"
        '<state_update>{"hp_change": -5, '
        '"inventory_add": [{"item_id": "torch_2", "item_name": "Torch", "quantity": 1}], '
        '"inventory_remove": ["potion_1"]}</state_update>'
    )
    cleaned = parse_and_apply_mutations(narrative, "kaelen", fake)

    assert "<state_update>" not in cleaned
    assert "bleeding into the murk" in cleaned
    assert fake.hp_updates == [("kaelen", -5)]
    assert fake.inventory_adds == [("torch_2", "kaelen", "Torch", 1)]
    assert fake.inventory_removes == [("potion_1", "kaelen")]
    print("[PASS] parse_and_apply_mutations applies hp/inventory mutations + cleans narrative")


def test_parse_and_apply_mutations_no_block_and_malformed() -> None:
    fake = FakeState(_sample_state("kaelen"))
    assert parse_and_apply_mutations("Plain prose.", "kaelen", fake) == "Plain prose."
    assert fake.hp_updates == [] and fake.inventory_adds == [] and fake.inventory_removes == []

    fake2 = FakeState(_sample_state("kaelen"))
    cleaned = parse_and_apply_mutations(
        "before <state_update>{not valid json</state_update> after", "kaelen", fake2
    )
    assert "<state_update>" not in cleaned
    assert "before" in cleaned and "after" in cleaned
    assert fake2.hp_updates == [] and fake2.inventory_adds == [] and fake2.inventory_removes == []
    print("[PASS] parser no-ops without a block; malformed JSON skipped but still cleaned")


def test_full_pipeline_applies_state_update_and_persists_chat() -> None:
    state = _sample_state("kaelen")
    fake_state = FakeState(state)
    fake_lore = FakeLore(_sample_lore())
    block = '<state_update>{"hp_change": -3}</state_update>'
    captured: dict[str, Any] = {}
    fake_llm = _llm_client_capturing(captured, narrative=CANNED_NARRATIVE + "\n\n" + block)

    _override(fake_state, fake_lore, fake_llm)
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/action",
            json={"player_id": "kaelen", "action_text": "I draw my blade and charge the Brotherhood warden."},
        )
    finally:
        _reset_overrides()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The block is stripped from the user-facing narrative.
    assert "<state_update>" not in body["narrative_response"]
    assert body["narrative_response"].endswith("slip away?")
    # The mutation went through the StateManager interface.
    assert fake_state.hp_updates == [("kaelen", -3)]
    # Both chat turns persisted, the user's action first.
    assert [role for (_p, role, _c) in fake_state.chat_turns] == ["user", "assistant"]
    assert fake_state.chat_turns[0][2] == "I draw my blade and charge the Brotherhood warden."
    print("[PASS] full pipeline: <state_update> applied, narrative cleaned, chat persisted")


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
    assert body["current_state"]["player"]["hp"] == 42
    assert body["current_state"]["player"]["level"] == 5

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
        test_build_messages_splices_chat_history,
        test_parse_and_apply_mutations_applies_and_cleans,
        test_parse_and_apply_mutations_no_block_and_malformed,
        test_full_pipeline_success,
        test_full_pipeline_applies_state_update_and_persists_chat,
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
