"""Core State Memory module.

SQL-backed store for hard game mechanics (stats, locations, NPCs, inventory).
Runs alongside the RAG/LoreManager system and is intentionally independent of it:
every fact that must be exact (HP, levels, quantities, who is where) lives in
SQLite instead of vector search.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS Players (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    hp                  INTEGER NOT NULL,
    max_hp              INTEGER NOT NULL,
    level               INTEGER NOT NULL DEFAULT 1,
    current_location_id TEXT REFERENCES Locations(id)
);

CREATE TABLE IF NOT EXISTS NPCs (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    location_id       TEXT REFERENCES Locations(id),
    disposition_score INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS Locations (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    base_description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS Inventory (
    item_id   TEXT PRIMARY KEY,
    owner_id  TEXT NOT NULL,
    item_name TEXT NOT NULL,
    quantity  INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_npcs_location ON NPCs (location_id);
CREATE INDEX IF NOT EXISTS idx_inventory_owner ON Inventory (owner_id);
"""


class StateManager:
    """Persistent SQLite store for hard game state, with per-call safe connections."""

    def __init__(self, db_path: str = "game_state.db") -> None:
        """Connect to `db_path` (creating the file if missing) and create all tables if absent."""
        self._db_path = db_path
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ upserts

    def upsert_player(
        self,
        player_id: str,
        name: str,
        hp: int,
        max_hp: int,
        level: int,
        current_location_id: Optional[str] = None,
    ) -> None:
        """Insert a Player row, or update name/hp/max_hp/level/location when the id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Players (id, name, hp, max_hp, level, current_location_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    hp = excluded.hp,
                    max_hp = excluded.max_hp,
                    level = excluded.level,
                    current_location_id = excluded.current_location_id
                """,
                (player_id, name, hp, max_hp, level, current_location_id),
            )

    def upsert_npc(
        self,
        npc_id: str,
        name: str,
        location_id: Optional[str],
        disposition_score: int = 0,
    ) -> None:
        """Insert an NPC row, or update name/location/disposition when the id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO NPCs (id, name, location_id, disposition_score)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    location_id = excluded.location_id,
                    disposition_score = excluded.disposition_score
                """,
                (npc_id, name, location_id, disposition_score),
            )

    def upsert_location(
        self,
        location_id: str,
        name: str,
        base_description: str = "",
    ) -> None:
        """Insert a Location row, or update name/base_description when the id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Locations (id, name, base_description)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    base_description = excluded.base_description
                """,
                (location_id, name, base_description),
            )

    def upsert_inventory_item(
        self,
        item_id: str,
        owner_id: str,
        item_name: str,
        quantity: int,
    ) -> None:
        """Insert an inventory row, or update owner/name/quantity when the item_id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Inventory (item_id, owner_id, item_name, quantity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    item_name = excluded.item_name,
                    quantity = excluded.quantity
                """,
                (item_id, owner_id, item_name, quantity),
            )

    # ---------------------------------------------------------------- retrieval

    def get_player_state(self, player_id: str) -> Optional[dict[str, Any]]:
        """Return one consolidated dict: player stats, current location, NPCs there, and inventory (None if the player is missing)."""
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row

            player = conn.execute(
                "SELECT id, name, hp, max_hp, level, current_location_id "
                "FROM Players WHERE id = ?",
                (player_id,),
            ).fetchone()
            if player is None:
                return None

            location: Optional[dict[str, Any]] = None
            npcs_here: list[dict[str, Any]] = []
            location_id = player["current_location_id"]
            if location_id is not None:
                loc = conn.execute(
                    "SELECT id, name, base_description FROM Locations WHERE id = ?",
                    (location_id,),
                ).fetchone()
                if loc is not None:
                    location = {
                        "id": loc["id"],
                        "name": loc["name"],
                        "base_description": loc["base_description"],
                    }
                npcs_here = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT id, name, disposition_score FROM NPCs "
                        "WHERE location_id = ? ORDER BY name",
                        (location_id,),
                    )
                ]

            inventory = [
                dict(row)
                for row in conn.execute(
                    "SELECT item_id, item_name, quantity FROM Inventory "
                    "WHERE owner_id = ? ORDER BY item_name",
                    (player_id,),
                )
            ]

        return {
            "player": {
                "id": player["id"],
                "name": player["name"],
                "hp": player["hp"],
                "max_hp": player["max_hp"],
                "level": player["level"],
                "current_location_id": location_id,
            },
            "location": location,
            "npcs_here": npcs_here,
            "inventory": inventory,
        }
