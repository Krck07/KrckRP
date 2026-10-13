"""Smoke test for the LoreManager insertion & storage module.

Exercises the real public API:
  - insert_lore() with the (text, lore_type, character_id, location_id, ...) signature
  - overlapping word-chunking of a long text
  - deterministic upsert behavior (re-inserting does not duplicate rows)
"""

from lore_manager import LoreManager

# Sample RPG Test Data, already mapped to the insert_lore() signature.
# `extra_metadata` carries fields that are not first-class params (e.g. faction_id).
TEST_LORE = [
    {
        "text": (
            "The Sunken Citadel of Oakhaven was constructed during the Second Age. "
            "It lies submerged deep within the Blackmist Swamps. Legend speaks of an elven relic "
            "hidden in the lower Sanctum, guarded by bound water elementals and ancient wards."
        ),
        "lore_type": "location",
        "location_id": "oakhaven_swamp",
        "source_id": "world_history_vol_2",
        "extra_metadata": None,
    },
    {
        "text": (
            "Master Alchemist Vaelen is a high elf exiled from Oakhaven centuries ago. "
            "He resides in the capital's lower ring, operating 'The Amber Flask'. "
            "Vaelen harbors a deep animosity toward the Obsidian Brotherhood, who cursed his former apprentice."
        ),
        "lore_type": "character_backstory",
        "character_id": "npc_vaelen",
        "location_id": "capital_lower_ring",
        "source_id": "npc_registry",
        "extra_metadata": None,
    },
    {
        "text": (
            "The Obsidian Brotherhood operates in secret throughout the realm. "
            "Recognized by their obsidian signet rings, members specialize in contraband alchemy, "
            "forbidden magic, and subterranean smuggling routes near Oakhaven."
        ),
        "lore_type": "faction",
        "location_id": None,
        "character_id": None,
        "source_id": "guild_records",
        "extra_metadata": {"faction_id": "obsidian_brotherhood"},
    },
]


def make_long_lore(word_count: int = 800) -> str:
    """Build a synthetic long text (~800 words) to force multi-chunk insertion."""
    sentence = (
        "The chronicle records that in the year of the Ashen Tide, the wardens of the "
        "northern gate swore oaths upon the cold iron of the Sunken Citadel."
    ).split()
    words = (sentence * (word_count // len(sentence) + 1))[:word_count]
    return " ".join(words)


def run_test() -> None:
    print("Initializing LoreManager...")
    lore_db = LoreManager()

    print("\n--- Test 1: Inserting short lore entries via insert_lore() ---")
    written_ids: list[str] = []
    for index, entry in enumerate(TEST_LORE, start=1):
        ids = lore_db.insert_lore(
            text=entry["text"],
            lore_type=entry["lore_type"],
            character_id=entry.get("character_id"),
            location_id=entry.get("location_id"),
            source_id=entry["source_id"],
            extra_metadata=entry["extra_metadata"],
        )
        written_ids.extend(ids)
        print(f"  [+] Inserted Entry {index} ({entry['lore_type']}) -> {len(ids)} chunk(s)")

    count_after_insert = lore_db.collection.count()
    print(f"  [i] Collection count after insertion: {count_after_insert}")

    print("\n--- Test 2: Verify stored documents + metadata ---")
    fetched = lore_db.collection.get(ids=written_ids, include=["documents", "metadatas"])
    for doc_id, doc, meta in zip(
        fetched["ids"], fetched["documents"], fetched["metadatas"]
    ):
        print(f"  [*] {doc_id}")
        print(f"      text: {doc[:70]}...")
        print(f"      meta: {meta}")

    print("\n--- Test 3: Long text is chunked with overlap ---")
    long_text = make_long_lore(word_count=800)
    long_ids = lore_db.insert_lore(
        text=long_text,
        lore_type="event",
        location_id="oakhaven_swamp",
        source_id="ashen_tide_chronicle",
    )
    long_fetch = lore_db.collection.get(ids=long_ids, include=["documents"])
    chunk_count = len(long_fetch["documents"])
    word_counts = [len(d.split()) for d in long_fetch["documents"]]
    print(f"  [+] 800-word text stored as {chunk_count} chunk(s), word sizes: {word_counts}")
    assert chunk_count > 1, "Expected the long text to be split into multiple chunks"
    assert max(word_counts) <= 250, "Every chunk must respect the 250-word window"

    # Verify overlap: chunk N+1 must begin with the tail words of chunk N.
    docs = long_fetch["documents"]
    tail = docs[0].split()[-50:]
    assert docs[1].split()[:50] == tail, "Consecutive chunks must share a 50-word overlap"
    print("  [+] Overlap verified: chunk 2 begins with the last 50 words of chunk 1")

    print("\n--- Test 4: Re-insertion is idempotent (upsert, no duplicates) ---")
    count_before = lore_db.collection.count()
    again_ids = lore_db.insert_lore(
        text=TEST_LORE[0]["text"],
        lore_type=TEST_LORE[0]["lore_type"],
        location_id=TEST_LORE[0]["location_id"],
        source_id=TEST_LORE[0]["source_id"],
    )
    count_after = lore_db.collection.count()
    assert again_ids == written_ids[:1] or again_ids[0] in written_ids, (
        "Same text must map to the same stable id"
    )
    assert count_after == count_before, "Re-inserting identical text must not add rows"
    print(f"  [+] Same id reused: {again_ids[0]}")
    print(f"  [+] Count unchanged: {count_before} -> {count_after}")

    print("\nAll insertion tests passed.")


if __name__ == "__main__":
    run_test()
