"""Lore insertion & storage module for the KrckRP roleplay engine.

This module handles the *write* side of the RAG memory pipeline:
chunking long lore texts and persisting them (with metadata) into a
ChromaDB collection. Retrieval / LLM querying is intentionally out of
scope for this session.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import chromadb
from chromadb.api import Collection
from chromadb.utils import embedding_functions


class LoreManager:
    """Persists world lore (character backstories, locations, events)
    into a ChromaDB vector collection so it can be retrieved later."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "world_lore",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ) -> None:
        """Set up the persistent ChromaDB client, the embedding function,
        and the target collection.

        The embedding model is placed on the best available accelerator,
        tried in priority order: CUDA GPU (only if present), DirectML
        (integrated graphics), then CPU. If every explicit device fails,
        it falls back to the SentenceTransformer library's own default
        device selection, so initialization never hard-crashes.

        Args:
            db_path: Filesystem location where ChromaDB persists data.
            collection_name: Name of the collection to get or create.
            embedding_model: SentenceTransformer model name used for
                embedding all documents and queries.
        """
        self._client: chromadb.ClientAPI = chromadb.PersistentClient(path=db_path)
        self._active_device: str = "unknown"
        self._embedding_function = self._create_embedding_function(embedding_model)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

    @property
    def collection(self) -> Collection:
        """Expose the underlying ChromaDB collection for advanced use."""
        return self._collection

    @property
    def device(self) -> str:
        """The device the embedding model was placed on
        (e.g. 'cuda', 'dml', 'cpu', or 'default' for library fallback)."""
        return self._active_device

    def _create_embedding_function(
        self, embedding_model: str
    ) -> embedding_functions.SentenceTransformerEmbeddingFunction:
        """Build the embedding function on the best available device.

        Candidate devices are tried in priority order — "cuda" (only when
        torch reports a GPU), "dml" (integrated graphics via DirectML),
        "cpu" — and every candidate is validated by actually loading the
        model and running one probe embedding, so a device string the
        installed torch build doesn't support can never slip through to
        runtime. If all explicit devices fail, the final candidate passes
        no ``device`` kwarg at all, letting the SentenceTransformer
        library choose its own default (the requested failsafe).

        Args:
            embedding_model: SentenceTransformer model name to load.

        Returns:
            A ready, verified SentenceTransformerEmbeddingFunction.

        Raises:
            RuntimeError: Only if even the library-default construction
                fails (e.g. model missing and no network access).
        """
        candidates: list[Optional[str]] = []
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                candidates.append("cuda")
        except ImportError:
            pass  # torch missing: skip explicit CUDA candidate
        candidates += ["dml", "cpu", None]  # None == library default (failsafe)

        last_error: Optional[BaseException] = None
        for device in candidates:
            try:
                kwargs: dict[str, Any] = {"device": device} if device is not None else {}
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model, **kwargs
                )
                # ChromaDB's wrapper loads the model lazily, so force the
                # load + one forward pass now to verify the device works.
                ef(["device probe"])
                self._active_device = device if device is not None else "default"
                print(f"[LoreManager] embedding device: {self._active_device}")
                return ef
            except BaseException as exc:  # noqa: BLE001 - deliberate broad catch
                last_error = exc

        raise RuntimeError(
            "Could not initialize the embedding model on any device "
            "(tried cuda, dml, cpu, library default). Last error: "
            f"{last_error}"
        )

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 250,
        overlap: int = 50,
    ) -> list[str]:
        """Split a long text into overlapping word-based chunks.

        Chunks are built in ``chunk_size`` word windows that slide forward
        by ``chunk_size - overlap`` words, so consecutive chunks share a
        small bridge of context. This keeps sentence boundaries from being
        cut mid-thought while still producing compact, vectorizable units.

        Args:
            text: The raw lore text to split.
            chunk_size: Target window size in words (recommended 200-300).
            overlap: Number of words shared between consecutive chunks.
                Must be strictly smaller than ``chunk_size``.

        Returns:
            A list of chunk strings in original order. Empty list if
            ``text`` contains no words.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")

        words: list[str] = text.split()
        if not words:
            return []

        step: int = chunk_size - overlap
        chunks: list[str] = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + chunk_size])
            chunks.append(chunk)
            # Stop early if the window no longer advances.
            if start + chunk_size >= len(words):
                break

        return chunks

    @staticmethod
    def _build_ids(source_id: str, chunks: list[str]) -> list[str]:
        """Derive stable, deterministic ChromaDB ids for a batch of chunks.

        Using content-hashed ids means re-inserting the same text updates
        the same rows (no duplicates), while different chunks of the same
        source stay grouped under a shared ``source_id`` prefix.
        """
        ids: list[str] = []
        for chunk in chunks:
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
            ids.append(f"{source_id}-{digest}")
        return ids

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str]:
        """Strip out keys with ``None`` values.

        ChromaDB rejects ``None`` as a metadata value, so unset optional
        fields (e.g. a location entry with no character_id) are dropped
        instead of being stored as nulls.
        """
        return {k: str(v) for k, v in metadata.items() if v is not None}

    def insert_lore(
        self,
        text: str,
        lore_type: str,
        character_id: Optional[str] = None,
        location_id: Optional[str] = None,
        source_id: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        chunk_size: int = 250,
        overlap: int = 50,
    ) -> list[str]:
        """Chunk a lore text and store every chunk in the collection.

        The text is split into overlapping word chunks, each chunk is
        embedded with the configured SentenceTransformer model, and the
        resulting vectors + metadata are written to ChromaDB under stable
        ids derived from the chunk content.

        Args:
            text: The lore content, e.g. a character backstory or a
                location description.
            lore_type: Categorization of the lore, e.g. "character",
                "location", "event", "item", "faction".
            character_id: Optional id of the character this lore belongs to.
            location_id: Optional id of the location this lore is about.
            source_id: Stable identifier for this logical source (e.g.
                "char_kael_01"). Defaults to a hash of the text, so the
                same text always maps to the same source.
            extra_metadata: Optional free-form metadata merged into every
                chunk (e.g. {"chapter": "3"}).
            chunk_size: Words per chunk (recommended 200-300).
            overlap: Overlap in words between consecutive chunks.

        Returns:
            The list of ChromaDB ids that were written (one per chunk),
            which can be used later for deletion or verification.

        Raises:
            ValueError: If ``text`` is empty or chunking parameters are
                invalid.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        ids = self._build_ids(
            source_id if source_id is not None else self._default_source_id(text),
            chunks,
        )

        base_metadata: dict[str, Any] = {
            "type_of_lore": lore_type,
            "character_id": character_id,
            "location_id": location_id,
            "source_id": source_id,
        }
        if extra_metadata:
            base_metadata.update(extra_metadata)
        base_metadata = self._sanitize_metadata(base_metadata)

        metadatas: list[dict[str, Any]] = []
        for index, _ in enumerate(chunks):
            metadata = dict(base_metadata)
            metadata["chunk_index"] = index
            metadata["total_chunks"] = len(chunks)
            metadatas.append(metadata)

        # upsert: re-inserting identical text updates the same rows
        # instead of creating duplicates.
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        return ids

    @staticmethod
    def _build_where_clause(metadata_filter: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Validate and normalize a metadata filter into a ChromaDB where clause.

        Simple equality dicts (``{"location_id": "oakhaven_swamp"}``) are passed
        straight through, as is ChromaDB's operator syntax (``{"key": {"$in": [...]}}``,
        ``{"$and": [...]}``). ``None`` values are stripped so the clause never
        contains nulls ChromaDB would reject.
        """
        if metadata_filter is None:
            return None
        if not isinstance(metadata_filter, dict) or not metadata_filter:
            raise ValueError("metadata_filter must be a non-empty dict if provided")
        sanitized: dict[str, Any] = {
            k: v for k, v in metadata_filter.items() if v is not None
        }
        if not sanitized:
            return None
        return sanitized

    def search_lore(
        self,
        query: str,
        n_results: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over the stored lore collection.

        Embeds the query with the same bge-base-en-v1.5 model used for
        insertion, runs ChromaDB's ``.query()`` with an optional metadata
        ``where`` clause, and flattens the nested result structure into a
        clean list of dictionaries.

        Args:
            query: Natural-language search text (e.g. "What do you know
                about the Obsidian Brotherhood?").
            n_results: Maximum number of chunks to return (default 3).
            metadata_filter: Optional equality/where-clause dict applied to
                the query, e.g. ``{"location_id": "oakhaven_swamp"}``.
                Only chunks whose stored metadata match are considered.

        Returns:
            A list (most similar first) of dicts, one per result:
            ``{"id": str, "text": str, "metadata": dict, "distance": float}``
            where a lower ``distance`` means higher similarity.

        Raises:
            ValueError: If ``query`` is empty, ``n_results`` is not
                positive, or ``metadata_filter`` is malformed.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if n_results <= 0:
            raise ValueError("n_results must be > 0")

        where_clause = self._build_where_clause(metadata_filter)

        raw = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB nests everything one level deep per query:
        # {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        ids: list[str] = raw.get("ids", [[]])[0]
        documents: list[Optional[str]] = raw.get("documents", [[]])[0] or []
        metadatas: list[Optional[dict[str, Any]]] = raw.get("metadatas", [[]])[0] or []
        distances: list[float] = raw.get("distances", [[]])[0] or []

        results: list[dict[str, Any]] = []
        for position, chunk_id in enumerate(ids):
            results.append(
                {
                    "id": chunk_id,
                    "text": documents[position] if position < len(documents) else "",
                    "metadata": metadatas[position]
                    if position < len(metadatas)
                    else {},
                    "distance": distances[position]
                    if position < len(distances)
                    else float("nan"),
                }
            )
        return results

    @staticmethod
    def _default_source_id(text: str) -> str:
        """Fallback source id derived from the text content."""
        return f"lore-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
