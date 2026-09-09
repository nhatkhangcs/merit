"""Transactional, provenance-aware memory for MERIT.

The persisted JSONL files are the human-readable source of truth.  The vector
index, row mapping, and metadata are derived artifacts, but normal loading is
strict: inconsistent state is rejected rather than silently repaired.  The
only repair entry point is :meth:`MemoryStore.rebuild_before_evaluation`.

Optional production dependencies (FAISS, NumPy, and sentence-transformers) are
imported lazily.  ``HashingEmbedder`` and ``PythonVectorIndex`` provide a fully
stdlib-backed path for deterministic unit tests and small smoke runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .config import ExperimentConfig, ModelConfig, RetrievalConfig, canonical_json, content_hash


_LOGGER = logging.getLogger(__name__)


POLARITIES = frozenset({"positive", "negative"})
APPLICABILITY_SCOPES = frozenset({"global", "same_database", "cross_database"})
MEMORY_FORMAT_VERSION = 3


class MemoryErrorBase(RuntimeError):
    """Base class for memory failures."""


class MemoryConsistencyError(MemoryErrorBase):
    """Raised when list, JSONL, metadata, mapping, or index state diverges."""


class EmbeddingDimensionError(MemoryErrorBase):
    """Raised when an embedding does not match the configured index dimension."""


class Embedder(Protocol):
    """Minimal embedding interface used by memory and retrieval."""

    @property
    def dimension(self) -> int:
        ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class VectorIndexBackend(Protocol):
    """Backend contract shared by FAISS and the stdlib implementation."""

    @property
    def dimension(self) -> int:
        ...

    @property
    def ntotal(self) -> int:
        ...

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        ...

    def scores(
        self,
        query_vector: Sequence[float],
        row_indices: Sequence[int],
    ) -> list[float]:
        ...

    def vectors(self) -> list[list[float]]:
        ...

    def save(self, path: str | Path) -> None:
        ...

    def load(self, path: str | Path) -> None:
        ...


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


_SQL_QUOTE_CLOSE = {"'": "'", '"': '"', "`": "`", "[": "]"}
_SQL_IDENTITY_PUNCTUATION = frozenset("(),=<>!+-*/")


def normalize_transformation(value: str) -> str:
    """Normalize SQL outside quoted regions while preserving literal contents."""

    text = value.strip()
    output: list[str] = []
    pending_space = False
    closing_quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]

        if closing_quote is not None:
            output.append(char)
            if char == "\\" and index + 1 < len(text):
                index += 1
                output.append(text[index])
            elif char == closing_quote:
                if index + 1 < len(text) and text[index + 1] == closing_quote:
                    index += 1
                    output.append(text[index])
                else:
                    closing_quote = None
            index += 1
            continue

        if char in _SQL_QUOTE_CLOSE:
            if pending_space and output and output[-1] not in _SQL_IDENTITY_PUNCTUATION:
                output.append(" ")
            closing_quote = _SQL_QUOTE_CLOSE[char]
            output.append(char)
            pending_space = False
        elif char.isspace():
            pending_space = True
        elif char in _SQL_IDENTITY_PUNCTUATION:
            if output and output[-1] == " ":
                output.pop()
            output.append(char)
            pending_space = False
        else:
            if pending_space and output and output[-1] not in _SQL_IDENTITY_PUNCTUATION:
                output.append(" ")
            output.append(char.lower())
            pending_space = False
        index += 1

    return "".join(output)


def memory_entry_id(attempted_sql_delta: str, error_type: str) -> str:
    """Return the deterministic ID shared by conflicting observations."""

    identity = {
        "error_type": _normalize_space(error_type),
        "transformation": normalize_transformation(attempted_sql_delta),
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def _observation_record(entry: "MemoryEntry") -> dict[str, Any]:
    return {
        "source_query_id": entry.source_query_id,
        "source_db_id": entry.source_db_id,
        "source_episode_id": entry.source_episode_id,
        "source_stream_position": entry.source_stream_position,
        "source_iteration": entry.source_iteration,
        "polarity": entry.polarity,
        "provenance": dict(entry.provenance),
        "question": entry.question,
        "schema": entry.schema,
        "failure_context": entry.failure_context,
        "attempted_sql_delta": entry.attempted_sql_delta,
        "successful_direction": entry.successful_direction,
        "observed_outcome": entry.observed_outcome,
        "observed_db_error": entry.observed_db_error,
    }


@dataclass(frozen=True)
class MemoryEntry:
    """One positive or negative observation with complete causal provenance."""

    source_query_id: str
    source_db_id: str
    source_episode_id: str
    source_stream_position: int
    source_iteration: int
    polarity: str
    provenance: Mapping[str, Any]
    failure_context: str
    error_type: str
    attempted_sql_delta: str
    observed_outcome: str
    observed_db_error: str | None
    applicability_scope: str
    question: str = ""
    schema: str = ""
    successful_direction: str = ""
    confirmation_count: int = 1
    success_count: int = 0
    failure_count: int = 0
    entry_id: str = ""
    provenance_history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("source_query_id", self.source_query_id),
                ("source_db_id", self.source_db_id),
                ("source_episode_id", self.source_episode_id),
                ("polarity", self.polarity),
                ("failure_context", self.failure_context),
                ("error_type", self.error_type),
                ("attempted_sql_delta", self.attempted_sql_delta),
                ("observed_outcome", self.observed_outcome),
                ("applicability_scope", self.applicability_scope),
            )
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(f"memory entry fields must be non-empty: {', '.join(missing)}")
        if self.polarity not in POLARITIES:
            raise ValueError(f"unsupported polarity: {self.polarity}")
        if self.applicability_scope not in APPLICABILITY_SCOPES:
            raise ValueError(
                "applicability_scope must be global, same_database, or cross_database"
            )
        if self.source_stream_position < 0 or self.source_iteration < 0:
            raise ValueError("source positions and iterations must be non-negative")
        if self.confirmation_count < 1:
            raise ValueError("confirmation_count must be positive")
        if self.success_count < 0 or self.failure_count < 0:
            raise ValueError("success/failure counts cannot be negative")
        if self.polarity == "positive" and not self.successful_direction.strip():
            raise ValueError("positive entries require successful_direction")
        if self.polarity == "positive" and self.observed_outcome != "CORRECT":
            raise ValueError("positive entries require a confirmed CORRECT outcome")
        if self.polarity == "negative" and self.observed_outcome == "CORRECT":
            raise ValueError("negative entries cannot carry a CORRECT outcome")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        try:
            canonical_json(dict(self.provenance))
        except (TypeError, ValueError) as exc:
            raise ValueError("provenance must be JSON serializable") from exc
        expected = memory_entry_id(self.attempted_sql_delta, self.error_type)
        if self.entry_id and self.entry_id != expected:
            raise ValueError("entry_id does not match transformation/error-type identity")

    def canonicalized(self) -> "MemoryEntry":
        """Validate and fill deterministic identity/evidence counters."""

        self.validate()
        success_count = self.success_count
        failure_count = self.failure_count
        if success_count == 0 and failure_count == 0:
            if self.polarity == "positive":
                success_count = self.confirmation_count
            else:
                failure_count = self.confirmation_count
        history = self.provenance_history or (_observation_record(self),)
        return replace(
            self,
            entry_id=memory_entry_id(self.attempted_sql_delta, self.error_type),
            success_count=success_count,
            failure_count=failure_count,
            confirmation_count=max(
                self.confirmation_count, success_count + failure_count
            ),
            provenance=dict(self.provenance),
            provenance_history=tuple(dict(item) for item in history),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = dict(self.provenance)
        payload["provenance_history"] = [
            dict(item) for item in self.provenance_history
        ]
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MemoryEntry":
        data = dict(raw)
        data["provenance"] = dict(data.get("provenance") or {})
        data["provenance_history"] = tuple(
            dict(item) for item in data.get("provenance_history") or ()
        )
        return cls(**data).canonicalized()


def _merge_entries(existing: MemoryEntry, incoming: MemoryEntry) -> MemoryEntry:
    if existing.entry_id != incoming.entry_id:
        raise ValueError("cannot merge different memory identities")

    success_count = existing.success_count + incoming.success_count
    failure_count = existing.failure_count + incoming.failure_count
    if success_count > failure_count:
        polarity = "positive"
    elif failure_count > success_count:
        polarity = "negative"
    else:
        polarity = existing.polarity

    histories = list(existing.provenance_history)
    seen = {canonical_json(item) for item in histories}
    for item in incoming.provenance_history:
        key = canonical_json(item)
        if key not in seen:
            histories.append(dict(item))
            seen.add(key)

    # Preserve both kinds of evidence.  Polarity only controls which retrieval
    # pool currently exposes the entry; counts retain any conflict.
    successful_direction = (
        incoming.successful_direction
        if incoming.successful_direction.strip()
        else existing.successful_direction
    )
    negative_source = incoming if incoming.polarity == "negative" else existing
    positive_source = incoming if incoming.polarity == "positive" else existing
    representative = positive_source if polarity == "positive" else negative_source

    return replace(
        representative,
        polarity=polarity,
        successful_direction=successful_direction,
        confirmation_count=success_count + failure_count,
        success_count=success_count,
        failure_count=failure_count,
        entry_id=existing.entry_id,
        provenance_history=tuple(histories),
    ).canonicalized()


def entry_embedding_text(entry: MemoryEntry) -> str:
    """Gold-free text embedded for memory similarity."""

    return " ".join(
        (
            f"QUESTION: {entry.question}",
            f"SCHEMA: {entry.schema}",
            f"FAILURE: {entry.failure_context}",
            f"TRANSFORMATION: {normalize_transformation(entry.attempted_sql_delta)}",
            f"ERROR: {entry.error_type}",
            f"OUTCOME: {entry.observed_outcome}",
            f"DB ERROR: {entry.observed_db_error or '(none)'}",
        )
    ).strip()


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


class HashingEmbedder:
    """Deterministic stdlib embedder intended for tests and smoke runs."""

    def __init__(self, dimension: int = 32):
        if dimension < 1:
            raise ValueError("embedding dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            vectors.append(_normalize_vector(vector))
        return vectors


class SentenceTransformerEmbedder:
    """Lazy production embedder pinned by ``ModelConfig``."""

    def __init__(self, model_name: str, revision: str):
        self.model_name = model_name
        self.revision = revision
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for production embeddings"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
            )
        return self._model

    @property
    def dimension(self) -> int:
        dimension = self._get_model().get_sentence_embedding_dimension()
        if not dimension:
            raise EmbeddingDimensionError("embedding model reported no dimension")
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]


class PythonVectorIndex:
    """Small deterministic vector index requiring only the Python stdlib."""

    backend_name = "python"

    def __init__(self, dimension: int):
        if dimension < 1:
            raise ValueError("index dimension must be positive")
        self._dimension = int(dimension)
        self._vectors: list[list[float]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        staged = []
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    f"index expects {self.dimension}, received {len(vector)}"
                )
            staged.append(_normalize_vector(vector))
        self._vectors = staged

    def scores(
        self,
        query_vector: Sequence[float],
        row_indices: Sequence[int],
    ) -> list[float]:
        if len(query_vector) != self.dimension:
            raise EmbeddingDimensionError(
                f"query has dimension {len(query_vector)}; expected {self.dimension}"
            )
        query = _normalize_vector(query_vector)
        scores = []
        for row in row_indices:
            if row < 0 or row >= self.ntotal:
                raise IndexError(f"vector row out of range: {row}")
            scores.append(sum(a * b for a, b in zip(query, self._vectors[row])))
        return scores

    def vectors(self) -> list[list[float]]:
        return [list(vector) for vector in self._vectors]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "backend": self.backend_name,
                    "dimension": self.dimension,
                    "vectors": self._vectors,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("backend") != self.backend_name:
            raise MemoryConsistencyError("persisted index backend is not python")
        if int(payload.get("dimension", -1)) != self.dimension:
            raise EmbeddingDimensionError("persisted Python index dimension mismatch")
        raw_vectors = payload.get("vectors")
        if not isinstance(raw_vectors, list):
            raise MemoryConsistencyError("persisted Python vectors must be a list")
        vectors: list[list[float]] = []
        for vector in raw_vectors:
            if not isinstance(vector, list) or len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    "persisted Python index vector dimension mismatch"
                )
            values = [float(value) for value in vector]
            if any(not math.isfinite(value) for value in values):
                raise MemoryConsistencyError("persisted Python vector is not finite")
            vectors.append(values)
        self._vectors = vectors


class FaissVectorIndex:
    """Production inner-product index; imports FAISS/NumPy only when used."""

    backend_name = "faiss"

    def __init__(self, dimension: int):
        if dimension < 1:
            raise ValueError("index dimension must be positive")
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss is required for the production index backend") from exc
        self._faiss = faiss
        self._dimension = int(dimension)
        self._index = faiss.IndexFlatIP(self.dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for the FAISS backend") from exc
        staged = [_normalize_vector(vector) for vector in vectors]
        if any(len(vector) != self.dimension for vector in staged):
            raise EmbeddingDimensionError("FAISS vector dimension mismatch")
        index = self._faiss.IndexFlatIP(self.dimension)
        if staged:
            index.add(np.asarray(staged, dtype="float32"))
        self._index = index

    def scores(
        self,
        query_vector: Sequence[float],
        row_indices: Sequence[int],
    ) -> list[float]:
        if len(query_vector) != self.dimension:
            raise EmbeddingDimensionError("FAISS query dimension mismatch")
        query = _normalize_vector(query_vector)
        scores: list[float] = []
        for row in row_indices:
            if row < 0 or row >= self.ntotal:
                raise IndexError(f"vector row out of range: {row}")
            vector = self._index.reconstruct(int(row))
            scores.append(sum(a * float(b) for a, b in zip(query, vector)))
        return scores

    def vectors(self) -> list[list[float]]:
        return [
            [float(value) for value in self._index.reconstruct(row)]
            for row in range(self.ntotal)
        ]

    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        index = self._faiss.read_index(str(path))
        if int(index.d) != self.dimension:
            raise EmbeddingDimensionError("persisted FAISS index dimension mismatch")
        self._index = index


BackendFactory = Callable[[int], VectorIndexBackend]


@dataclass(frozen=True)
class MemoryCounters:
    embedding_calls: int
    embedded_texts: int


class MemoryStore:
    """A strict transactional store for one deduplicated memory collection."""

    def __init__(
        self,
        directory: str | Path,
        config: ExperimentConfig | RetrievalConfig,
        *,
        embedder: Embedder | None = None,
        model_config: ModelConfig | None = None,
        backend_factory: BackendFactory | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if isinstance(config, ExperimentConfig):
            self.retrieval_config = config.retrieval
            resolved_model = config.model
        else:
            self.retrieval_config = config
            resolved_model = model_config or ModelConfig()
        self.retrieval_config.validate()

        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(
            resolved_model.embedding_name,
            resolved_model.embedding_revision,
        )
        self._dimension = int(self.embedder.dimension)
        if self._dimension < 1:
            raise EmbeddingDimensionError("embedding dimension must be positive")

        if backend_factory is None:
            index_class: type[PythonVectorIndex] | type[FaissVectorIndex]
            index_class = (
                FaissVectorIndex
                if self.retrieval_config.index_backend == "faiss"
                else PythonVectorIndex
            )
            backend_factory = index_class
        self._backend_factory = backend_factory

        self.positive_path = self.directory / "memory_positive.jsonl"
        self.negative_path = self.directory / "memory_negative.jsonl"
        suffix = (
            "faiss"
            if self.retrieval_config.index_backend == "faiss"
            else "json"
        )
        self.index_path = self.directory / f"memory_index.{suffix}"
        self.row_mapping_path = self.directory / "memory_row_to_entry_id.json"
        self.metadata_path = self.directory / "memory_metadata.json"
        self._artifact_paths = (
            self.positive_path,
            self.negative_path,
            self.index_path,
            self.row_mapping_path,
            self.metadata_path,
        )

        self._entries: list[MemoryEntry] = []
        self._entry_by_id: dict[str, MemoryEntry] = {}
        self._row_by_id: dict[str, int] = {}
        self._vectors: list[list[float]] = []
        self._index = self._new_index([])
        self._embedding_calls = 0
        self._embedded_texts = 0
        self._load_or_initialize()

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    @property
    def positive_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(entry for entry in self._entries if entry.polarity == "positive")

    @property
    def negative_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(entry for entry in self._entries if entry.polarity == "negative")

    @property
    def counters(self) -> MemoryCounters:
        return MemoryCounters(self._embedding_calls, self._embedded_texts)

    @property
    def embedding_calls(self) -> int:
        return self._embedding_calls

    @property
    def embedded_texts(self) -> int:
        return self._embedded_texts

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def index_ntotal(self) -> int:
        return self._index.ntotal

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entry_by_id.get(entry_id)

    def list_entries(self, polarity: str | None = None) -> tuple[MemoryEntry, ...]:
        if polarity is None:
            return self.entries
        if polarity not in POLARITIES:
            raise ValueError(f"unsupported polarity: {polarity}")
        return tuple(entry for entry in self._entries if entry.polarity == polarity)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = self.embedder.encode(list(texts))
        vectors = [_normalize_vector(vector) for vector in raw]
        if len(vectors) != len(texts):
            raise EmbeddingDimensionError("embedder returned the wrong number of vectors")
        for vector in vectors:
            if len(vector) != self.embedding_dimension:
                raise EmbeddingDimensionError(
                    f"embedder returned dimension {len(vector)}; "
                    f"expected {self.embedding_dimension}"
                )
        self._embedding_calls += 1
        self._embedded_texts += len(texts)
        return vectors

    def dense_scores(
        self,
        query_vector: Sequence[float],
        entry_ids: Sequence[str],
    ) -> list[float]:
        rows = []
        for entry_id in entry_ids:
            try:
                rows.append(self._row_by_id[entry_id])
            except KeyError as exc:
                raise KeyError(f"unknown memory entry: {entry_id}") from exc
        return self._index.scores(query_vector, rows)

    def add_entry(self, candidate: MemoryEntry) -> MemoryEntry:
        """Validate, embed, stage, and atomically commit one observation.

        Duplicate transformation/error-type identities merge evidence counts.
        Any failure rolls disk and in-memory state back and is re-raised.
        """

        old_embedding_calls = self._embedding_calls
        old_embedded_texts = self._embedded_texts
        try:
            incoming = candidate.canonicalized()
            existing = self._entry_by_id.get(incoming.entry_id)
            staged_entry = (
                _merge_entries(existing, incoming) if existing else incoming
            )

            # Embedding happens before any list, index, or disk mutation.
            vector = self.embed_texts([entry_embedding_text(staged_entry)])[0]
            if len(vector) != self.embedding_dimension:
                raise EmbeddingDimensionError("candidate embedding dimension mismatch")

            staged_entries = list(self._entries)
            staged_vectors = [list(item) for item in self._vectors]
            if existing is None:
                staged_entries.append(staged_entry)
                staged_vectors.append(vector)
            else:
                row = self._row_by_id[existing.entry_id]
                staged_entries[row] = staged_entry
                staged_vectors[row] = vector

            staged_index = self._new_index(staged_vectors)
            self._commit_snapshot(staged_entries, staged_vectors, staged_index)
            return self._entry_by_id[staged_entry.entry_id]
        except Exception:
            # Counters are observable run state and participate in rollback.
            self._embedding_calls = old_embedding_calls
            self._embedded_texts = old_embedded_texts
            raise

    def assert_consistent(self) -> None:
        """Assert all in-memory and persisted cardinality/identity invariants."""

        entry_count = len(self._entries)
        if len(self._entry_by_id) != entry_count:
            raise MemoryConsistencyError("entry ID map cardinality mismatch")
        if len(self._row_by_id) != entry_count:
            raise MemoryConsistencyError("row map cardinality mismatch")
        if len(self._vectors) != entry_count:
            raise MemoryConsistencyError("cached vector cardinality mismatch")
        if self._index.ntotal != entry_count:
            raise MemoryConsistencyError(
                f"entries/index mismatch: {entry_count} != {self._index.ntotal}"
            )
        if self._index.dimension != self.embedding_dimension:
            raise MemoryConsistencyError("index/embedding dimension mismatch")
        persisted_index = self._backend_factory(self.embedding_dimension)
        persisted_index.load(self.index_path)
        if persisted_index.ntotal != entry_count:
            raise MemoryConsistencyError(
                "entries/persisted-index count mismatch"
            )
        if persisted_index.dimension != self.embedding_dimension:
            raise MemoryConsistencyError("persisted index dimension mismatch")
        persisted_vectors = persisted_index.vectors()


        positive_raw = self._read_jsonl(self.positive_path)
        negative_raw = self._read_jsonl(self.negative_path)
        jsonl_count = len(positive_raw) + len(negative_raw)
        if jsonl_count != entry_count:
            raise MemoryConsistencyError(
                f"entries/JSONL mismatch: {entry_count} != {jsonl_count}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if int(metadata.get("entry_count", -1)) != entry_count:
            raise MemoryConsistencyError("entries/metadata count mismatch")
        expected_positive = sum(entry.polarity == "positive" for entry in self._entries)
        if int(metadata.get("positive_count", -1)) != expected_positive:
            raise MemoryConsistencyError("positive/metadata count mismatch")
        expected_negative = sum(entry.polarity == "negative" for entry in self._entries)
        if int(metadata.get("negative_count", -1)) != expected_negative:
            raise MemoryConsistencyError("negative/metadata count mismatch")
        if int(metadata.get("format_version", -1)) != MEMORY_FORMAT_VERSION:
            raise MemoryConsistencyError("unsupported memory format version")
        if int(metadata.get("embedding_dimension", -1)) != self.embedding_dimension:
            raise MemoryConsistencyError("metadata embedding dimension mismatch")
        if metadata.get("index_backend") != self.retrieval_config.index_backend:
            raise MemoryConsistencyError("metadata index backend mismatch")
        persisted_vector_hash = content_hash(persisted_vectors)
        if metadata.get("vector_hash") != persisted_vector_hash:
            raise MemoryConsistencyError("persisted index vector hash mismatch")
        if content_hash(self._vectors) != persisted_vector_hash:
            raise MemoryConsistencyError("in-memory/persisted vector mismatch")

        with self.row_mapping_path.open("r", encoding="utf-8") as handle:
            row_mapping = json.load(handle)
        expected_ids = [entry.entry_id for entry in self._entries]
        if row_mapping != expected_ids:
            raise MemoryConsistencyError("persisted row-to-entry-ID mapping mismatch")
        if list(self._row_by_id) != expected_ids:
            raise MemoryConsistencyError("in-memory row-to-entry-ID mapping mismatch")

        persisted = positive_raw + negative_raw
        persisted_by_id = {
            str(item.get("entry_id")): MemoryEntry.from_dict(item)
            for item in persisted
        }
        if len(persisted_by_id) != entry_count:
            raise MemoryConsistencyError("duplicate IDs found in JSONL")
        for entry in self._entries:
            if persisted_by_id.get(entry.entry_id) != entry:
                raise MemoryConsistencyError(
                    f"persisted entry differs from in-memory entry: {entry.entry_id}"
                )

    @classmethod
    def rebuild_before_evaluation(
        cls,
        directory: str | Path,
        config: ExperimentConfig | RetrievalConfig,
        *,
        embedder: Embedder | None = None,
        model_config: ModelConfig | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> "MemoryStore":
        """Explicitly rebuild derived artifacts from JSONL before evaluation.

        This is deliberately not an automatic recovery path.  JSONL must parse,
        contain valid deterministic IDs, and contain no duplicate identities.
        """

        root = Path(directory)
        positive_path = root / "memory_positive.jsonl"
        negative_path = root / "memory_negative.jsonl"
        if not positive_path.is_file() or not negative_path.is_file():
            raise MemoryConsistencyError(
                "pre-evaluation rebuild requires both polarity JSONL files"
            )
        raw_entries: list[MemoryEntry] = []
        for path, expected_polarity in (
            (positive_path, "positive"),
            (negative_path, "negative"),
        ):
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        entry = MemoryEntry.from_dict(json.loads(line))
                    except Exception as exc:
                        raise MemoryConsistencyError(
                            f"invalid {path.name}:{line_number}"
                        ) from exc
                    if entry.polarity != expected_polarity:
                        raise MemoryConsistencyError(
                            f"{path.name} contains {entry.polarity} entry"
                        )
                    raw_entries.append(entry)
        if len({entry.entry_id for entry in raw_entries}) != len(raw_entries):
            raise MemoryConsistencyError("cannot rebuild duplicate JSONL identities")

        # Construct in a temporary directory so normal strict loading never sees
        # the corrupted target state.
        with tempfile.TemporaryDirectory(
            prefix=".memory-rebuild-", dir=str(root.parent)
        ) as temp_dir:
            temp_root = Path(temp_dir)
            rebuilt = cls(
                temp_root,
                config,
                embedder=embedder,
                model_config=model_config,
                backend_factory=backend_factory,
            )
            for entry in raw_entries:
                rebuilt.add_entry(entry)
            rebuilt.assert_consistent()

            target_artifacts = [
                positive_path,
                negative_path,
                root / rebuilt.index_path.name,
                root / "memory_row_to_entry_id.json",
                root / "memory_metadata.json",
            ]
            source_artifacts = [
                rebuilt.positive_path,
                rebuilt.negative_path,
                rebuilt.index_path,
                rebuilt.row_mapping_path,
                rebuilt.metadata_path,
            ]
            for source, target in zip(source_artifacts, target_artifacts):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)

        return cls(
            root,
            config,
            embedder=embedder,
            model_config=model_config,
            backend_factory=backend_factory,
        )

    def _new_index(
        self,
        vectors: Sequence[Sequence[float]],
    ) -> VectorIndexBackend:
        index = self._backend_factory(self.embedding_dimension)
        index.build(vectors)
        if index.dimension != self.embedding_dimension:
            raise EmbeddingDimensionError("backend created the wrong dimension")
        if index.ntotal != len(vectors):
            raise MemoryConsistencyError("staged index cardinality mismatch")
        return index

    def _load_or_initialize(self) -> None:
        present = [path.is_file() for path in self._artifact_paths]
        if not any(present):
            self._commit_snapshot([], [], self._new_index([]))
            return
        if not all(present):
            missing = [
                path.name
                for path, exists in zip(self._artifact_paths, present)
                if not exists
            ]
            raise MemoryConsistencyError(
                "incomplete persisted memory; explicit pre-evaluation rebuild "
                f"is required (missing: {', '.join(missing)})"
            )

        positive = [
            MemoryEntry.from_dict(item)
            for item in self._read_jsonl(self.positive_path)
        ]
        negative = [
            MemoryEntry.from_dict(item)
            for item in self._read_jsonl(self.negative_path)
        ]
        if any(entry.polarity != "positive" for entry in positive):
            raise MemoryConsistencyError("positive JSONL contains wrong polarity")
        if any(entry.polarity != "negative" for entry in negative):
            raise MemoryConsistencyError("negative JSONL contains wrong polarity")

        with self.row_mapping_path.open("r", encoding="utf-8") as handle:
            row_ids = json.load(handle)
        by_id = {entry.entry_id: entry for entry in positive + negative}
        if len(by_id) != len(positive) + len(negative):
            raise MemoryConsistencyError("duplicate persisted memory IDs")
        try:
            entries = [by_id[entry_id] for entry_id in row_ids]
        except KeyError as exc:
            raise MemoryConsistencyError("row mapping references an unknown entry") from exc
        if len(entries) != len(by_id):
            raise MemoryConsistencyError("row mapping does not cover every entry")

        index = self._backend_factory(self.embedding_dimension)
        index.load(self.index_path)
        vectors = index.vectors()
        self._install_snapshot(entries, vectors, index)
        self.assert_consistent()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise MemoryConsistencyError(
                        f"invalid JSON in {path.name}:{line_number}"
                    ) from exc
        return rows

    def _metadata(
        self, entries: Sequence[MemoryEntry], vectors: Sequence[Sequence[float]]
    ) -> dict[str, Any]:
        return {
            "format_version": MEMORY_FORMAT_VERSION,
            "entry_count": len(entries),
            "positive_count": sum(
                entry.polarity == "positive" for entry in entries
            ),
            "negative_count": sum(
                entry.polarity == "negative" for entry in entries
            ),
            "embedding_dimension": self.embedding_dimension,
            "index_backend": self.retrieval_config.index_backend,
            "vector_hash": content_hash(vectors),
        }

    @staticmethod
    def _jsonl_bytes(entries: Iterable[MemoryEntry]) -> bytes:
        text = "".join(
            json.dumps(
                entry.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for entry in entries
        )
        return text.encode("utf-8")

    def _commit_snapshot(
        self,
        entries: Sequence[MemoryEntry],
        vectors: Sequence[Sequence[float]],
        index: VectorIndexBackend,
    ) -> None:
        entries = list(entries)
        vectors = [list(vector) for vector in vectors]
        if len(entries) != len(vectors) or len(entries) != index.ntotal:
            raise MemoryConsistencyError("cannot stage inconsistent memory snapshot")
        index_vectors = index.vectors()
        if len(index_vectors) != len(entries) or any(
            len(vector) != self.embedding_dimension for vector in index_vectors
        ):
            raise MemoryConsistencyError("staged index vectors are inconsistent")
        ids = [entry.entry_id for entry in entries]
        if len(set(ids)) != len(ids):
            raise MemoryConsistencyError("cannot stage duplicate entry IDs")

        old_entries = self._entries
        old_vectors = self._vectors
        old_index = self._index
        token = next(tempfile._get_candidate_names())
        staged_paths = {
            path: path.with_name(f".{path.name}.{token}.staged")
            for path in self._artifact_paths
        }
        backup_paths = {
            path: path.with_name(f".{path.name}.{token}.backup")
            for path in self._artifact_paths
        }
        replaced: list[Path] = []
        backed_up: list[Path] = []

        try:
            positive = [entry for entry in entries if entry.polarity == "positive"]
            negative = [entry for entry in entries if entry.polarity == "negative"]
            staged_paths[self.positive_path].write_bytes(self._jsonl_bytes(positive))
            staged_paths[self.negative_path].write_bytes(self._jsonl_bytes(negative))
            staged_paths[self.row_mapping_path].write_text(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            staged_paths[self.metadata_path].write_text(
                json.dumps(
                    self._metadata(entries, index_vectors),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            index.save(staged_paths[self.index_path])

            for path in self._artifact_paths:
                if path.exists():
                    os.replace(path, backup_paths[path])
                    backed_up.append(path)
            for path in self._artifact_paths:
                os.replace(staged_paths[path], path)
                replaced.append(path)

            self._install_snapshot(entries, index_vectors, index)
            self.assert_consistent()
        except Exception:
            self._install_snapshot(old_entries, old_vectors, old_index)
            for path in replaced:
                if path.exists():
                    path.unlink()
            for path in reversed(backed_up):
                backup = backup_paths[path]
                if backup.exists():
                    os.replace(backup, path)
            for staged in staged_paths.values():
                if staged.exists():
                    staged.unlink()
            raise
        else:
            for backup in backup_paths.values():
                try:
                    backup.unlink(missing_ok=True)
                except OSError as error:
                    _LOGGER.warning(
                        "memory snapshot committed but backup cleanup failed for %s: %s",
                        backup,
                        error,
                    )

    def _install_snapshot(
        self,
        entries: Sequence[MemoryEntry],
        vectors: Sequence[Sequence[float]],
        index: VectorIndexBackend,
    ) -> None:
        self._entries = list(entries)
        self._vectors = [list(vector) for vector in vectors]
        self._index = index
        self._entry_by_id = {entry.entry_id: entry for entry in self._entries}
        self._row_by_id = {
            entry.entry_id: row for row, entry in enumerate(self._entries)
        }


__all__ = [
    "APPLICABILITY_SCOPES",
    "EmbeddingDimensionError",
    "FaissVectorIndex",
    "HashingEmbedder",
    "MemoryConsistencyError",
    "MemoryCounters",
    "MemoryEntry",
    "MemoryStore",
    "POLARITIES",
    "PythonVectorIndex",
    "SentenceTransformerEmbedder",
    "entry_embedding_text",
    "memory_entry_id",
    "normalize_transformation",
]
