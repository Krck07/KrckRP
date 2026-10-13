"""Retrieval / semantic search tests for the LoreManager.

Runs against the data inserted by test_insertion.py:
  3 short entries (oakhaven location, Vaelen backstory, Obsidian Brotherhood)
  + a 4-chunk 800-word "Ashen Tide" chronicle tagged location_id=oakhaven_swamp.

Query 1: general lore question about the Obsidian Brotherhood (no filter)
Query 2: character query about Vaelen (no filter)
Query 3: strict metadata filter for location_id == "oakhaven_swamp"
"""

from typing import Any

from lore_manager import LoreManager


def print_results(results: list[dict[str, Any]]) -> None:
    """Pretty-print a flattened search_lore() result list."""
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        tags = ", ".join(
            f"{k}={v}" for k, v in meta.items() if k not in ("chunk_index", "total_chunks")
        )
        print(f"  [{rank}] distance={result['distance']:.4f}  ({tags})")
        print(f"      text: {result['text'][:90]}...")


def run_test() -> None:
    print("Initializing LoreManager...")
    lore_db = LoreManager()
    print(f"Collection contains {lore_db.collection.count()} chunks.\n")

    # --- Query 1: general lore question, no metadata filter ---
    print("--- Query 1: General lore question about the Obsidian Brotherhood ---")
    q1 = lore_db.search_lore(
        query="Who is the Obsidian Brotherhood and what are they known for?",
        n_results=3,
    )
    print_results(q1)
    assert len(q1) > 0, "Expected at least one result"
    assert any(
        result["metadata"].get("type_of_lore") == "faction" for result in q1
    ), "Expected the faction entry to rank in the top results"
    print("  [PASS] Faction lore retrieved\n")

    # --- Query 2: character query about Vaelen ---
    print("--- Query 2: Character query about Vaelen ---")
    q2 = lore_db.search_lore(
        query="Tell me about the alchemist Vaelen and his past.",
        n_results=3,
    )
    print_results(q2)
    assert len(q2) > 0, "Expected at least one result"
    assert any(
        result["metadata"].get("character_id") == "npc_vaelen" for result in q2
    ), "Expected Vaelen's backstory to rank in the top results"
    print("  [PASS] Vaelen backstory retrieved\n")

    # --- Query 3: strict metadata filter on location_id ---
    print("--- Query 3: Strict metadata filter {location_id: oakhaven_swamp} ---")
    q3 = lore_db.search_lore(
        query="What can be found in the swamps?",
        n_results=5,
        metadata_filter={"location_id": "oakhaven_swamp"},
    )
    print_results(q3)
    assert len(q3) > 0, "Expected at least one result matching the filter"
    assert all(
        result["metadata"].get("location_id") == "oakhaven_swamp" for result in q3
    ), "Every result must satisfy the metadata filter"
    print("  [PASS] All results satisfy location_id == oakhaven_swamp\n")

    # --- Query 4 (bonus): result ordering must be by ascending distance ---
    print("--- Query 4: Sanity check - results sorted by ascending distance ---")
    assert q3 == sorted(q3, key=lambda r: r["distance"]), "Results must be most-similar-first"
    print("  [PASS] Distances are in ascending order\n")

    print("All retrieval tests passed.")


if __name__ == "__main__":
    run_test()
