"""KrckRP world ingestion pipeline.

Standalone CLI that loads a campaign template (``world_data.json`` by default,
or any path passed as the first positional argument) and populates BOTH memory
stores:

    * SQLite  — locations, NPCs, starter inventory items, quests
                (via :class:`state_memory.state_manager.StateManager`)
    * ChromaDB — chunked, embedded lore
                (via :class:`lore_manager.LoreManager`)

Usage
-----
    python world_builder.py                      # uses ./world_data.json
    python world_builder.py path/to/my_world.json

Design notes
------------
* The heavy managers are imported **lazily** inside :func:`main` (types are
  ``TYPE_CHECKING``-only), so importing this module stays cheap and never
  loads chromadb/torch.
* Ingestion order is dependency-safe: locations first (NPCs and lore reference
  them), then NPCs, items, quests, and lore last (the embedding step).
* Bad *entries* are skipped with a warning (one typo shouldn't abort a whole
  campaign load); bad *structure* (file missing, invalid JSON, top-level not an
  object, a section that is not a list of objects) fails loudly with exit 1.
* Re-running is safe: every write is an upsert (SQLite ``ON CONFLICT`` /
  ChromaDB content-hashed ids), so the same template can be re-ingested
  idempotently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # runtime imports happen lazily in main(); see module docstring
    from lore_manager import LoreManager
    from state_memory.state_manager import StateManager

log = logging.getLogger("world_builder")

DEFAULT_WORLD_FILE = "world_data.json"

# Same env-overridable store configuration as orchestrator.py, so both tools
# target the same SQLite file and ChromaDB collection by default.
STATE_DB_PATH: str = os.environ.get("STATE_DB_PATH", "game_state.db")
CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION: str = os.environ.get("CHROMA_COLLECTION", "world_lore")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Top-level sections that must be a list of objects when present.
LIST_SECTIONS: tuple[str, ...] = ("locations", "npcs", "lore", "quests", "items", "abilities")


# --------------------------------------------------------------------------- #
# Field helpers (strict, with actionable error messages)
# --------------------------------------------------------------------------- #
def _require_str(entry: dict[str, Any], key: str, context: str) -> str:
    """Return ``entry[key]`` as a non-blank string, or raise a clear ValueError."""
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: required string field '{key}' missing or empty")
    return value.strip()


def _optional_str(entry: dict[str, Any], key: str) -> Optional[str]:
    """Return ``entry[key]`` when it is a non-blank string, else ``None``."""
    value = entry.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError(f"field '{key}' must be a string, got {type(value).__name__}")
    return value.strip()


def _optional_int(entry: dict[str, Any], key: str, default: int) -> int:
    """Return ``entry[key]`` as a positive int, or ``default`` when absent."""
    value = entry.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"field '{key}' must be a positive integer, got {value!r}")
    return value


def _optional_signed_int(entry: dict[str, Any], key: str, default: int = 0) -> int:
    """Return ``entry[key]`` as any int (signed), or ``default`` when absent.

    For fields that may legitimately be negative (e.g. NPC disposition).
    """
    value = entry.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field '{key}' must be an integer, got {value!r}")
    return value


def _section_entries(
    data: dict[str, Any],
    section: str,
    label: str,
) -> list[tuple[int, dict[str, Any]]]:
    """Collect the dict entries of a top-level section as (index, entry) pairs.

    Absent sections yield ``[]`` (an empty world section is legal). Non-dict
    entries are skipped with a warning. Raises ValueError when the section is
    present but not a list (validated up-front in build_world, but checked
    here too for direct calls).
    """
    if section not in data:
        return []

    entries = data[section]
    if not isinstance(entries, list):
        raise ValueError(f"section '{section}' must be a JSON array of objects")

    collected: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            log.warning("%s[%d]: expected an object, got %s — skipped",
                        label, index, type(entry).__name__)
            continue
        collected.append((index, entry))
    return collected


# --------------------------------------------------------------------------- #
# Section ingesters (one per store/table)
# --------------------------------------------------------------------------- #
def ingest_locations(
    state_mgr: "StateManager",
    entries: Sequence[tuple[int, dict[str, Any]]],
) -> int:
    """Ingest the ``locations`` section via :meth:`StateManager.upsert_location`.

    Each entry: ``{"location_id": "...", "name": "...", "description": "..."}``
    (``description`` optional). Returns the number of locations written.
    """
    written = 0
    for index, entry in entries:
        context = f"locations[{index}]"
        try:
            location_id = _require_str(entry, "location_id", context)
            name = _require_str(entry, "name", context)
            description = _optional_str(entry, "description") or ""
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        state_mgr.upsert_location(location_id=location_id, name=name, base_description=description)
        written += 1
    return written


def ingest_npcs(
    state_mgr: "StateManager",
    entries: Sequence[tuple[int, dict[str, Any]]],
) -> int:
    """Ingest the ``npcs`` section via :meth:`StateManager.upsert_npc`.

    Each entry: ``{"npc_id": "...", "name": "...", "location_id": "...",
    "disposition_score": 0}`` (``location_id`` / ``disposition_score``
    optional). Returns the number of NPCs written.
    """
    written = 0
    for index, entry in entries:
        context = f"npcs[{index}]"
        try:
            npc_id = _require_str(entry, "npc_id", context)
            name = _require_str(entry, "name", context)
            location_id = _optional_str(entry, "location_id")
            disposition = _optional_signed_int(entry, "disposition_score", 0)
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        state_mgr.upsert_npc(
            npc_id=npc_id, name=name, location_id=location_id, disposition_score=disposition
        )
        written += 1
    return written


def ingest_items(
    state_mgr: "StateManager",
    entries: Sequence[tuple[int, dict[str, Any]]],
    default_owner_id: Optional[str],
) -> int:
    """Ingest the ``items`` section via :meth:`StateManager.upsert_inventory_item`.

    Each entry: ``{"item_id": "...", "item_name": "...", "owner_id": "...",
    "quantity": 1}`` (``owner_id`` falls back to the top-level
    ``default_player_id``; ``quantity`` defaults to 1). Returns the number of
    items written.
    """
    written = 0
    for index, entry in entries:
        context = f"items[{index}]"
        try:
            item_id = _require_str(entry, "item_id", context)
            item_name = _require_str(entry, "item_name", context)
            owner_id = _optional_str(entry, "owner_id") or default_owner_id
            quantity = _optional_int(entry, "quantity", 1)
            if owner_id is None:
                raise ValueError(f"{context}: no owner_id (set entry 'owner_id' "
                                 "or top-level 'default_player_id')")
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        state_mgr.upsert_inventory_item(
            item_id=item_id, owner_id=owner_id, item_name=item_name, quantity=quantity
        )
        written += 1
    return written


def ingest_quests(
    state_mgr: "StateManager",
    entries: Sequence[tuple[int, dict[str, Any]]],
    default_player_id: Optional[str],
) -> int:
    """Ingest the ``quests`` section via :meth:`StateManager.upsert_quest`.

    Each entry: ``{"quest_id": "...", "title": "...", "description": "...",
    "status": "active", "player_id": "..."}`` (``description`` / ``status`` /
    ``player_id`` optional; ``status`` defaults to ``"active"``, ``player_id``
    falls back to the top-level ``default_player_id``). Returns the number of
    quests written.
    """
    written = 0
    for index, entry in entries:
        context = f"quests[{index}]"
        try:
            quest_id = _require_str(entry, "quest_id", context)
            title = _require_str(entry, "title", context)
            description = _optional_str(entry, "description") or ""
            status = _optional_str(entry, "status") or "active"
            player_id = _optional_str(entry, "player_id") or default_player_id
            if player_id is None:
                raise ValueError(f"{context}: no player_id (set entry 'player_id' "
                                 "or top-level 'default_player_id')")
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        state_mgr.upsert_quest(
            player_id=player_id, quest_id=quest_id, title=title, description=description, status=status
        )
        written += 1
    return written


def ingest_abilities(
    state_mgr: "StateManager",
    entries: Sequence[tuple[int, dict[str, Any]]],
    default_player_id: Optional[str],
) -> int:
    """Ingest the ``abilities`` section via :meth:`StateManager.upsert_ability`.

    Each entry: ``{"ability_id": "...", "name": "...", "resource_cost": 10,
    "player_id": "..."}`` (``resource_cost`` defaults to 0, ``player_id``
    falls back to ``default_player_id``). Returns the number of abilities written.
    """
    written = 0
    for index, entry in entries:
        context = f"abilities[{index}]"
        try:
            ability_id = _require_str(entry, "ability_id", context)
            name = _require_str(entry, "name", context)
            player_id = _optional_str(entry, "player_id") or default_player_id
            resource_cost = _optional_signed_int(entry, "resource_cost", 0)
            if player_id is None:
                raise ValueError(f"{context}: no player_id (set entry 'player_id' or top-level 'default_player_id')")
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        state_mgr.upsert_ability(
            player_id=player_id, ability_id=ability_id, name=name, resource_cost=resource_cost
        )
        written += 1
    return written


def ingest_lore(
    lore_mgr: "LoreManager",
    entries: Sequence[tuple[int, dict[str, Any]]],
) -> int:
    """Ingest the ``lore`` section via :meth:`LoreManager.insert_lore`.

    Each entry: ``{"text": "...", "lore_type": "faction",
    "location_id": "...", "character_id": "...", "source_id": "...",
    "chunk_size": 250, "overlap": 50}`` (all optional except ``text``;
    ``lore_type`` defaults to ``"general"``). Returns the total number of
    CHROMADB CHUNKS written across all entries (a long entry produces several).
    """
    chunks = 0
    for index, entry in entries:
        context = f"lore[{index}]"
        try:
            text = _require_str(entry, "text", context)
            lore_type = _optional_str(entry, "lore_type") or "general"
            location_id = _optional_str(entry, "location_id")
            character_id = _optional_str(entry, "character_id")
            source_id = _optional_str(entry, "source_id")
            chunk_size = _optional_int(entry, "chunk_size", 250)
            overlap = _optional_int(entry, "overlap", 50)
            if overlap >= chunk_size:
                raise ValueError(f"{context}: 'overlap' ({overlap}) must be < 'chunk_size' ({chunk_size})")
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        ids = lore_mgr.insert_lore(
            text=text,
            lore_type=lore_type,
            character_id=character_id,
            location_id=location_id,
            source_id=source_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks += len(ids)
        log.info("%s: %d chunk(s) embedded (source_id=%s)", context, len(ids), source_id or "auto")
    return chunks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _default_player_id(data: dict[str, Any]) -> Optional[str]:
    """Top-level ``default_player_id`` (optional) used as owner/quest fallback."""
    value = data.get("default_player_id")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError("'default_player_id' must be a string")
    return value.strip()


def build_world(
    data: dict[str, Any],
    state_mgr: "StateManager",
    lore_mgr: "LoreManager",
) -> dict[str, int]:
    """Validate ``data`` and ingest every section into both stores.

    Ingestion order is dependency-safe: locations -> npcs -> items -> quests
    (SQLite), then lore (ChromaDB embedding — the heavy step, last).

    Returns a per-section summary dict, e.g.
    ``{"locations": 3, "npcs": 4, "items": 6, "quests": 2, "lore_chunks": 9}``.
    """
    if not isinstance(data, dict):
        raise ValueError("world_data.json must be a single JSON object")
    for section in LIST_SECTIONS:
        if section in data and not isinstance(data[section], list):
            raise ValueError(f"section '{section}' must be a JSON array of objects")

    default_player = _default_player_id(data)

    summary: dict[str, int] = {
        "locations": ingest_locations(state_mgr, _section_entries(data, "locations", "locations")),
        "npcs": ingest_npcs(state_mgr, _section_entries(data, "npcs", "npcs")),
        "items": ingest_items(
            state_mgr, _section_entries(data, "items", "items"), default_player
        ),
        "quests": ingest_quests(
            state_mgr, _section_entries(data, "quests", "quests"), default_player
        ),
        "abilities": ingest_abilities(
            state_mgr, _section_entries(data, "abilities", "abilities"), default_player
        ),
    }
    summary["lore_chunks"] = ingest_lore(
        lore_mgr, _section_entries(data, "lore", "lore")
    )
    return summary


def _load_world_file(path: Path) -> dict[str, Any]:
    """Read + parse the campaign template; fail loudly on missing/invalid input."""
    if not path.is_file():
        raise FileNotFoundError(f"world data file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a single JSON object")
    return data


def _print_summary(path: Path, summary: dict[str, int]) -> None:
    print(f"[world_builder] Ingested {path}:")
    for key, value in summary.items():
        print(f"  {key:<12} {value}")
    print("[world_builder] Done. SQLite hard state + ChromaDB lore are up to date.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: ``python world_builder.py [path/to/world_data.json]``."""
    parser = argparse.ArgumentParser(
        prog="world_builder",
        description=(
            "KrckRP world ingestion: load a world_data.json campaign template and "
            "populate the SQLite state DB (locations, NPCs, items, quests) and the "
            "ChromaDB lore collection (chunked + embedded)."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_WORLD_FILE,
        help=f"Path to the world data JSON file (default: {DEFAULT_WORLD_FILE})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    path = Path(args.path)
    try:
        data = _load_world_file(path)

        # Lazy imports: chromadb/torch are only loaded when ingestion actually runs.
        from state_memory.state_manager import StateManager
        from lore_manager import LoreManager

        state_mgr = StateManager(db_path=STATE_DB_PATH)
        lore_mgr = LoreManager(
            db_path=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
        )
        summary = build_world(data, state_mgr, lore_mgr)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - report any store/embedding failure cleanly
        log.error("Ingestion failed: %s: %s", type(exc).__name__, exc)
        return 1

    _print_summary(path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
