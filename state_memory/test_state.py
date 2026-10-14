"""Standalone smoke test for StateManager.

Usage:
    python state_memory/test_state.py
"""

import json
import os

from state_manager import StateManager


def main() -> None:
    # Keep the test database next to this module so it doesn't pollute the repo root.
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_state.db")
    state = StateManager(db_path=db_path)

    # 1) Starting location
    state.upsert_location(
        location_id="loc_01",
        name="Emberfall Tavern",
        base_description="A soot-stained inn at the edge of the marsh road. "
        "Candlelight gutters behind rain-fogged glass and the air smells of roasting meat.",
    )

    # 2) Test player (also exercises the upsert path twice)
    state.upsert_player(
        player_id="player_01",
        name="Kaelen",
        hp=24,
        max_hp=30,
        level=3,
        current_location_id="loc_01",
    )
    state.upsert_player(  # update in place: lost 2 HP to a trap, gained a level
        player_id="player_01",
        name="Kaelen",
        hp=22,
        max_hp=30,
        level=4,
        current_location_id="loc_01",
    )

    # 3) NPC sharing the player's location
    state.upsert_npc(
        npc_id="npc_01",
        name="Berta the Innkeeper",
        location_id="loc_01",
        disposition_score=15,
    )

    # 4) Two inventory items (the potion quantity is also upserted)
    state.upsert_inventory_item("item_01", "player_01", "Iron Longsword", 1)
    state.upsert_inventory_item("item_02", "player_01", "Healing Potion", 1)
    state.upsert_inventory_item("item_02", "player_01", "Healing Potion", 3)

    # 5) Consolidated retrieval
    snapshot = state.get_player_state("player_01")
    print(json.dumps(snapshot, indent=2))

    assert snapshot is not None, "Player should exist after upserts"
    assert snapshot["player"]["hp"] == 22, "Upsert should have updated hp"
    assert snapshot["player"]["level"] == 4, "Upsert should have updated level"
    assert snapshot["location"]["name"] == "Emberfall Tavern"
    assert len(snapshot["npcs_here"]) == 1
    assert snapshot["npcs_here"][0]["name"] == "Berta the Innkeeper"
    assert {i["item_name"]: i["quantity"] for i in snapshot["inventory"]} == {
        "Iron Longsword": 1,
        "Healing Potion": 3,
    }
    assert state.get_player_state("no_such_player") is None
    print("\nAll assertions passed.")
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    main()
