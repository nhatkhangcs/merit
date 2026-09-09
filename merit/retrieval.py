"""Causal typed hybrid retrieval for MERIT and its controlled variants."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baselines import (
    HIGH_PRECISION_RETRIEVAL_TYPES,
    METHOD_POLICIES,
    get_method_policy,
)
from .config import ExperimentConfig, RetrievalConfig
from .memory import MemoryEntry, MemoryStore, POLARITIES


MEMORYLESS_VARIANTS = frozenset({"zeroshot", "iterative", "vanilla", "reflexion"})
TYPED_VARIANTS = frozenset(
    {
        "merit",
        "merit_full",
        "positive_only",
        "no_dense_rerank",
        "no_bm25",
        "random_same_type",
        "cross_database_only",
        "transductive_batch",
    }
)
CONDITIONALLY_TYPED_VARIANTS = frozenset(
    name
    for name, policy in METHOD_POLICIES.items()
    if policy.hard_filter_error_types is not None
)
UNTYPED_VARIANTS = frozenset({"no_type_filter", "dynamic_rag"})
SUPPORTED_VARIANTS = (
    MEMORYLESS_VARIANTS
    | TYPED_VARIANTS
    | CONDITIONALLY_TYPED_VARIANTS
    | UNTYPED_VARIANTS
)
CONFIDENCE_AWARE_VARIANT = "confidence_aware_type_filter"
_LOG_ARRAY_FIELDS = (
    "retrieved_entry_ids",
    "retrieved_polarities",
    "retrieved_error_types",
    "source_db_ids",
    "source_stream_positions",
    "dense_scores",
    "bm25_scores",
    "final_scores",
)
_CONFIDENCE_LOG_FIELDS = (
    "type_filter_applied",
    "type_match_used_as_soft_score",
    "type_match_scores",
)



@dataclass(frozen=True)
class RetrievalLog:
    """Complete retrieval provenance for one repair step."""

    query_id: str
    repair_iteration: int
    current_error_type: str
    legal_pool_size: int
    typed_pool_size: int
    fallback_used: bool
    type_filter_applied: bool
    type_match_used_as_soft_score: bool
    retrieved_entry_ids: tuple[str, ...]
    retrieved_polarities: tuple[str, ...]
    retrieved_error_types: tuple[str, ...]
    source_db_ids: tuple[str, ...]
    source_stream_positions: tuple[int, ...]
    dense_scores: tuple[float, ...]
    bm25_scores: tuple[float, ...]
    type_match_scores: tuple[float, ...]
    final_scores: tuple[float, ...]
    method_name: str
    legal_pool_size_by_polarity: dict[str, int]
    typed_pool_size_by_polarity: dict[str, int]
    fallback_used_by_polarity: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    positive_entries: tuple[MemoryEntry, ...]
    negative_entries: tuple[MemoryEntry, ...]
    log: RetrievalLog

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        by_id = {
            entry.entry_id: entry
            for entry in self.positive_entries + self.negative_entries
        }
        return tuple(by_id[entry_id] for entry_id in self.log.retrieved_entry_ids)


@dataclass(frozen=True)
class _Pool:
    polarity: str
    legal: tuple[MemoryEntry, ...]
    typed: tuple[MemoryEntry, ...]
    candidates: tuple[MemoryEntry, ...]
    fallback_used: bool


@dataclass(frozen=True)
class _Ranked:
    entry: MemoryEntry
    dense_score: float
    bm25_score: float
    type_match_score: float
    final_score: float


@dataclass(frozen=True)
class _ExpectedRetrievalLog:
    query_id: str
    repair_iteration: int
    current_error_type: str
    retrieved_entry_ids: tuple[str, ...]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _entry_document(entry: MemoryEntry) -> str:
    return " ".join(
        (
            entry.question,
            entry.failure_context,
            entry.attempted_sql_delta,
            entry.error_type,
            entry.observed_outcome,
            entry.observed_db_error or "",
            entry.successful_direction,
        )
    )


def _bm25_scores(query: str, entries: Sequence[MemoryEntry]) -> list[float]:
    """Compute BM25 over exactly the supplied candidate pool."""

    if not entries:
        return []
    query_tokens = _tokens(query)
    documents = [_tokens(_entry_document(entry)) for entry in entries]
    document_count = len(documents)
    average_length = (
        sum(len(document) for document in documents) / document_count
        if document_count
        else 0.0
    )
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if frequency == 0:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            length_norm = (
                1.0 - b + b * len(document) / average_length
                if average_length
                else 1.0
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0)
                / (frequency + k1 * length_norm)
            )
        scores.append(score)
    return scores


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        value = 0.0 if math.isclose(high, 0.0, abs_tol=1e-12) else 1.0
        return [value] * len(scores)
    width = high - low
    return [(float(score) - low) / width for score in scores]


def _expected_retrieval_logs(
    trajectories: Sequence[Mapping[str, Any]],
    method_name: str,
) -> tuple[_ExpectedRetrievalLog, ...]:
    if not get_method_policy(method_name).uses_global_memory:
        return ()

    expected: list[_ExpectedRetrievalLog] = []
    for trajectory in trajectories:
        query_id = str(trajectory.get("query_id", "")).strip()
        repair_steps = trajectory.get("repair_steps")
        retrieved_by_iteration = trajectory.get("retrieved_entry_ids")
        classifications = trajectory.get("classifications")
        if not query_id:
            raise ValueError("trajectory query_id must be non-empty")
        if (
            isinstance(repair_steps, bool)
            or not isinstance(repair_steps, int)
            or repair_steps < 0
        ):
            raise ValueError(f"{query_id}: repair_steps must be non-negative")
        if (
            not isinstance(retrieved_by_iteration, list)
            or len(retrieved_by_iteration) != repair_steps
        ):
            raise ValueError(
                f"{query_id}: retrieved_entry_ids do not align with repair steps"
            )
        if repair_steps and (
            not isinstance(classifications, list)
            or len(classifications) < repair_steps
        ):
            raise ValueError(
                f"{query_id}: classifications do not cover every retrieval"
            )

        for repair_iteration, entry_ids in enumerate(
            retrieved_by_iteration, start=1
        ):
            if not isinstance(entry_ids, list) or not all(
                isinstance(entry_id, str) for entry_id in entry_ids
            ):
                raise ValueError(
                    f"{query_id}: retrieved entry IDs must be a list of strings"
                )
            classification = classifications[repair_iteration - 1]
            if classification is None:
                current_error_type = "Unknown"
            elif isinstance(classification, Mapping):
                current_error_type = str(
                    classification.get("error_type", "")
                ).strip()
                if not current_error_type:
                    raise ValueError(
                        f"{query_id}: classification error_type is missing"
                    )
            else:
                raise ValueError(
                    f"{query_id}: classification must be an object or null"
                )
            expected.append(
                _ExpectedRetrievalLog(
                    query_id=query_id,
                    repair_iteration=repair_iteration,
                    current_error_type=current_error_type,
                    retrieved_entry_ids=tuple(entry_ids),
                )
            )
    return tuple(expected)


def retrieval_log_expected_count(
    trajectories: Sequence[Mapping[str, Any]],
    method_name: str,
) -> int:
    return len(_expected_retrieval_logs(trajectories, method_name))


def _finite_scores(
    values: Any,
    *,
    field: str,
    row_number: int,
) -> tuple[float, ...]:
    if not isinstance(values, list):
        raise ValueError(
            f"retrieval log row {row_number} field {field} must be an array"
        )
    scores: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"retrieval log row {row_number} field {field} "
                "must contain finite numbers"
            )
        scores.append(float(value))
    return tuple(scores)


def validate_retrieval_log_alignment(
    logs: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    method_name: str,
    *,
    type_match_weight: float = 0.10,
) -> None:
    expected = _expected_retrieval_logs(trajectories, method_name)
    if len(logs) != len(expected):
        raise ValueError(
            "retrieval log count does not align with committed repair steps"
        )

    confidence_aware = method_name == CONFIDENCE_AWARE_VARIANT
    for row_number, (row, expected_row) in enumerate(
        zip(logs, expected), start=1
    ):
        if not isinstance(row, Mapping):
            raise ValueError(f"retrieval log row {row_number} must be an object")
        if str(row.get("query_id", "")).strip() != expected_row.query_id:
            raise ValueError(
                f"retrieval log row {row_number} query_id does not align"
            )
        repair_iteration = row.get("repair_iteration")
        if (
            isinstance(repair_iteration, bool)
            or repair_iteration != expected_row.repair_iteration
        ):
            raise ValueError(
                f"retrieval log row {row_number} repair_iteration does not align"
            )
        if row.get("method_name") != method_name:
            raise ValueError(
                f"retrieval log row {row_number} method_name does not align"
            )
        if row.get("current_error_type") != expected_row.current_error_type:
            raise ValueError(
                f"retrieval log row {row_number} current_error_type does not align"
            )

        missing_arrays = [field for field in _LOG_ARRAY_FIELDS if field not in row]
        if missing_arrays:
            raise ValueError(
                f"retrieval log row {row_number} is missing arrays: "
                + ", ".join(missing_arrays)
            )
        entry_ids = row["retrieved_entry_ids"]
        if not isinstance(entry_ids, list) or not all(
            isinstance(entry_id, str) for entry_id in entry_ids
        ):
            raise ValueError(
                f"retrieval log row {row_number} retrieved_entry_ids "
                "must be an array of strings"
            )
        if tuple(entry_ids) != expected_row.retrieved_entry_ids:
            raise ValueError(
                f"retrieval log row {row_number} retrieved_entry_ids "
                "do not match the trajectory"
            )
        selected_count = len(entry_ids)
        for field in _LOG_ARRAY_FIELDS[1:]:
            values = row[field]
            if not isinstance(values, list) or len(values) != selected_count:
                raise ValueError(
                    f"retrieval log row {row_number} field {field} "
                    "does not align with retrieved entries"
                )

        dense_scores = _finite_scores(
            row["dense_scores"], field="dense_scores", row_number=row_number
        )
        bm25_scores = _finite_scores(
            row["bm25_scores"], field="bm25_scores", row_number=row_number
        )
        final_scores = _finite_scores(
            row["final_scores"], field="final_scores", row_number=row_number
        )
        present_confidence_fields = [
            field for field in _CONFIDENCE_LOG_FIELDS if field in row
        ]
        if confidence_aware and len(present_confidence_fields) != len(
            _CONFIDENCE_LOG_FIELDS
        ):
            raise ValueError(
                f"retrieval log row {row_number} is missing "
                "confidence-aware fields"
            )
        if present_confidence_fields and len(present_confidence_fields) != len(
            _CONFIDENCE_LOG_FIELDS
        ):
            raise ValueError(
                f"retrieval log row {row_number} has a partial "
                "confidence-aware schema"
            )
        if not confidence_aware:
            continue

        type_filter_applied = row["type_filter_applied"]
        soft_score_used = row["type_match_used_as_soft_score"]
        if not isinstance(type_filter_applied, bool) or not isinstance(
            soft_score_used, bool
        ):
            raise ValueError(
                f"retrieval log row {row_number} confidence flags "
                "must be boolean"
            )
        expected_hard_filter = (
            expected_row.current_error_type in HIGH_PRECISION_RETRIEVAL_TYPES
        )
        if (
            type_filter_applied != expected_hard_filter
            or soft_score_used == expected_hard_filter
        ):
            raise ValueError(
                f"retrieval log row {row_number} confidence policy differs"
            )
        type_match_scores = _finite_scores(
            row["type_match_scores"],
            field="type_match_scores",
            row_number=row_number,
        )
        if len(type_match_scores) != selected_count:
            raise ValueError(
                f"retrieval log row {row_number} type_match_scores "
                "do not align with retrieved entries"
            )
        retrieved_error_types = row["retrieved_error_types"]
        expected_type_scores = tuple(
            1.0
            if soft_score_used
            and retrieved_error_type == expected_row.current_error_type
            else 0.0
            for retrieved_error_type in retrieved_error_types
        )
        if type_match_scores != expected_type_scores:
            raise ValueError(
                f"retrieval log row {row_number} type-match scores differ"
            )
        for dense, bm25, type_match, final in zip(
            dense_scores,
            bm25_scores,
            type_match_scores,
            final_scores,
        ):
            recomputed = round(
                0.75 * dense + 0.25 * bm25 + type_match_weight * type_match,
                12,
            )
            if not math.isclose(
                final, recomputed, rel_tol=0.0, abs_tol=1e-10
            ):
                raise ValueError(
                    f"retrieval log row {row_number} final score differs "
                    "from confidence-aware components"
                )


class HybridRetriever:
    """Legal filtering followed by the selected method's candidate policy.

    Full MERIT retrieves each polarity independently and defaults to at most
    three positives and one negative.  If a same-type pool has fewer than
    ``min_typed_pool`` entries, the only documented fallback is the legal pool
    of that same polarity.  The confidence-aware variant keeps this hard filter
    for Syntax, Schema Linking, and Execution.  For every other error type it
    scores the full legal same-polarity pool with normalized dense/BM25 scores
    plus the configured 0.10 binary type-match bonus.  No score is computed
    before the candidate-pool decision, and no type bonus follows hard filtering.
    """

    def __init__(
        self,
        memory: MemoryStore,
        config: ExperimentConfig | RetrievalConfig,
        *,
        method_name: str | None = None,
        random_seed: int | None = None,
        log_path: str | Path | None = None,
    ):
        self.memory = memory
        if isinstance(config, ExperimentConfig):
            self.config = config.retrieval
            self.method_name = method_name or config.method_name
            self.random_seed = (
                config.random_seed if random_seed is None else random_seed
            )
        else:
            self.config = config
            self.method_name = method_name or "merit_full"
            self.random_seed = 42 if random_seed is None else random_seed
        self.config.validate()
        if self.method_name not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported retrieval variant: {self.method_name}")
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs: list[RetrievalLog] = []
        self.retrieval_calls = 0

    def retrieve(
        self,
        *,
        query_id: str,
        db_id: str,
        stream_position: int,
        repair_iteration: int,
        current_error_type: str,
        question: str,
        current_sql: str = "",
        failure_context: str = "",
        method_name: str | None = None,
        allowed_source_db_ids: Iterable[str] | None = None,
        disallowed_source_db_ids: Iterable[str] = (),
    ) -> RetrievalResult:
        """Retrieve legal cases for one repair step and append a full log."""

        variant = method_name or self.method_name
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported retrieval variant: {variant}")
        policy = get_method_policy(variant)
        type_filter_applied = policy.applies_hard_type_filter(current_error_type)
        type_match_used_as_soft_score = (
            variant in CONDITIONALLY_TYPED_VARIANTS and not type_filter_applied
        )
        if not query_id.strip() or not db_id.strip():
            raise ValueError("query_id and db_id must be non-empty")
        if stream_position < 0 or repair_iteration < 0:
            raise ValueError("stream position and repair iteration must be non-negative")

        self.retrieval_calls += 1
        query_text = " ".join(
            (
                question,
                current_sql,
                failure_context,
                current_error_type,
            )
        ).strip()
        allowed = set(allowed_source_db_ids) if allowed_source_db_ids is not None else None
        disallowed = set(disallowed_source_db_ids)
        cross_database_only = variant == "cross_database_only"

        if variant in MEMORYLESS_VARIANTS:
            return self._finish(
                query_id=query_id,
                repair_iteration=repair_iteration,
                current_error_type=current_error_type,
                variant=variant,
                pools=(),
                ranked=(),
                type_filter_applied=type_filter_applied,
                type_match_used_as_soft_score=type_match_used_as_soft_score,
            )

        if variant == "dynamic_rag":
            legal = tuple(
                entry
                for entry in self.memory.entries
                if self._is_legal(
                    entry,
                    query_id=query_id,
                    db_id=db_id,
                    stream_position=stream_position,
                    polarity=None,
                    allowed_source_db_ids=allowed,
                    disallowed_source_db_ids=disallowed,
                    cross_database_only=cross_database_only,
                )
            )
            pool = _Pool("combined", legal, (), legal, False)
            ranked = self._rank(
                pool.candidates,
                query_text=query_text,
                query_id=query_id,
                repair_iteration=repair_iteration,
                polarity="combined",
                variant=variant,
                query_vector=None,
                limit=self.config.max_positive + self.config.max_negative,
                current_error_type=current_error_type,
                type_match_used_as_soft_score=type_match_used_as_soft_score,
            )
            return self._finish(
                query_id=query_id,
                repair_iteration=repair_iteration,
                current_error_type=current_error_type,
                variant=variant,
                pools=(pool,),
                ranked=ranked,
                type_filter_applied=type_filter_applied,
                type_match_used_as_soft_score=type_match_used_as_soft_score,
            )

        causal = variant != "transductive_batch"
        polarities = ("positive",) if variant == "positive_only" else (
            "positive",
            "negative",
        )
        pools = tuple(
            self._pool(
                polarity=polarity,
                query_id=query_id,
                db_id=db_id,
                stream_position=stream_position,
                current_error_type=current_error_type,
                allowed_source_db_ids=allowed,
                disallowed_source_db_ids=disallowed,
                cross_database_only=cross_database_only,
                use_type_filter=type_filter_applied,
                causal=causal,
            )
            for polarity in polarities
        )

        dense_enabled = variant != "no_dense_rerank" and variant != "random_same_type"
        query_vector = None
        if dense_enabled and any(pool.candidates for pool in pools):
            query_vector = self.memory.embed_texts([query_text])[0]

        ranked: list[_Ranked] = []
        for pool in pools:
            limit = (
                self.config.max_positive
                if pool.polarity == "positive"
                else self.config.max_negative
            )
            selected = self._rank(
                pool.candidates,
                query_text=query_text,
                query_id=query_id,
                repair_iteration=repair_iteration,
                polarity=pool.polarity,
                variant=variant,
                query_vector=query_vector,
                limit=limit,
                current_error_type=current_error_type,
                type_match_used_as_soft_score=type_match_used_as_soft_score,
            )
            if type_filter_applied and not pool.fallback_used:
                if any(
                    item.entry.error_type != current_error_type
                    for item in selected
                ):
                    raise AssertionError(
                        "typed retrieval returned a mismatched error type without fallback"
                    )
            ranked.extend(selected)

        return self._finish(
            query_id=query_id,
            repair_iteration=repair_iteration,
            current_error_type=current_error_type,
            variant=variant,
            pools=pools,
            ranked=ranked,
            type_filter_applied=type_filter_applied,
            type_match_used_as_soft_score=type_match_used_as_soft_score,
        )

    def _pool(
        self,
        *,
        polarity: str,
        query_id: str,
        db_id: str,
        stream_position: int,
        current_error_type: str,
        allowed_source_db_ids: set[str] | None,
        disallowed_source_db_ids: set[str],
        cross_database_only: bool,
        use_type_filter: bool,
        causal: bool,
    ) -> _Pool:
        if polarity not in POLARITIES:
            raise ValueError(f"unsupported polarity: {polarity}")
        # Wrong-polarity entries are removed here, before the typed pool exists.
        legal = tuple(
            entry
            for entry in self.memory.entries
            if self._is_legal(
                entry,
                query_id=query_id,
                db_id=db_id,
                stream_position=stream_position,
                polarity=polarity,
                allowed_source_db_ids=allowed_source_db_ids,
                disallowed_source_db_ids=disallowed_source_db_ids,
                cross_database_only=cross_database_only,
                causal=causal,
            )
        )
        typed = tuple(
            entry for entry in legal if entry.error_type == current_error_type
        )
        if not use_type_filter:
            return _Pool(polarity, legal, typed, legal, False)
        if len(typed) >= self.config.min_typed_pool:
            return _Pool(polarity, legal, typed, typed, False)
        # The sole documented fallback is legal entries of this same polarity.
        return _Pool(polarity, legal, typed, legal, True)

    @staticmethod
    def _is_legal(
        entry: MemoryEntry,
        *,
        query_id: str,
        db_id: str,
        stream_position: int,
        polarity: str | None,
        allowed_source_db_ids: set[str] | None,
        disallowed_source_db_ids: set[str],
        cross_database_only: bool,
        causal: bool = True,
    ) -> bool:
        if causal:
            if entry.source_stream_position >= stream_position:
                return False
            if entry.source_query_id == query_id:
                return False
        if polarity is not None and entry.polarity != polarity:
            return False
        if allowed_source_db_ids is not None and (
            entry.source_db_id not in allowed_source_db_ids
        ):
            return False
        if entry.source_db_id in disallowed_source_db_ids:
            return False
        if cross_database_only and entry.source_db_id == db_id:
            return False
        if entry.applicability_scope == "same_database" and (
            entry.source_db_id != db_id
        ):
            return False
        if entry.applicability_scope == "cross_database" and (
            entry.source_db_id == db_id
        ):
            return False
        return True

    def _rank(
        self,
        candidates: Sequence[MemoryEntry],
        *,
        query_text: str,
        query_id: str,
        repair_iteration: int,
        polarity: str,
        variant: str,
        query_vector: Sequence[float] | None,
        limit: int,
        current_error_type: str,
        type_match_used_as_soft_score: bool,
    ) -> tuple[_Ranked, ...]:
        if not candidates or limit <= 0:
            return ()

        if variant == "random_same_type":
            seed_payload = (
                f"{self.random_seed}|{query_id}|{repair_iteration}|{polarity}"
            )
            seed = int.from_bytes(
                hashlib.sha256(seed_payload.encode("utf-8")).digest()[:8],
                "big",
            )
            shuffled = list(candidates)
            random.Random(seed).shuffle(shuffled)
            return tuple(
                _Ranked(entry, 0.0, 0.0, 0.0, 0.0)
                for entry in shuffled[:limit]
            )

        dense_enabled = variant != "no_dense_rerank"
        bm25_enabled = variant != "no_bm25"
        if dense_enabled:
            if query_vector is None:
                query_vector = self.memory.embed_texts([query_text])[0]
            dense_raw = self.memory.dense_scores(
                query_vector,
                [entry.entry_id for entry in candidates],
            )
            dense = _normalize_scores(dense_raw)
        else:
            dense = [0.0] * len(candidates)

        if bm25_enabled:
            bm25 = _normalize_scores(_bm25_scores(query_text, candidates))
        else:
            bm25 = [0.0] * len(candidates)
        type_match = [
            1.0
            if type_match_used_as_soft_score
            and entry.error_type == current_error_type
            else 0.0
            for entry in candidates
        ]

        dense_weight = self.config.dense_weight if dense_enabled else 0.0
        bm25_weight = self.config.bm25_weight if bm25_enabled else 0.0
        total_weight = dense_weight + bm25_weight
        if total_weight <= 0:
            raise ValueError("retrieval variant disabled every scoring component")
        dense_weight /= total_weight
        bm25_weight /= total_weight

        ranked = [
            _Ranked(
                entry=entry,
                dense_score=round(dense_score, 12),
                bm25_score=round(bm25_score, 12),
                type_match_score=type_match_score,
                final_score=round(
                    dense_weight * dense_score
                    + bm25_weight * bm25_score
                    + self.config.type_match_weight * type_match_score,
                    12,
                ),
            )
            for entry, dense_score, bm25_score, type_match_score in zip(
                candidates, dense, bm25, type_match
            )
        ]
        ranked.sort(
            key=lambda item: (
                -item.final_score,
                -item.dense_score,
                -item.bm25_score,
                -item.type_match_score,
                item.entry.entry_id,
            )
        )
        return tuple(ranked[:limit])

    def _finish(
        self,
        *,
        query_id: str,
        repair_iteration: int,
        current_error_type: str,
        variant: str,
        pools: Sequence[_Pool],
        ranked: Sequence[_Ranked],
        type_filter_applied: bool,
        type_match_used_as_soft_score: bool,
    ) -> RetrievalResult:
        entries = tuple(item.entry for item in ranked)
        log = RetrievalLog(
            query_id=query_id,
            repair_iteration=repair_iteration,
            current_error_type=current_error_type,
            legal_pool_size=sum(len(pool.legal) for pool in pools),
            typed_pool_size=sum(len(pool.typed) for pool in pools),
            fallback_used=any(pool.fallback_used for pool in pools),
            type_filter_applied=type_filter_applied,
            type_match_used_as_soft_score=type_match_used_as_soft_score,
            retrieved_entry_ids=tuple(entry.entry_id for entry in entries),
            retrieved_polarities=tuple(entry.polarity for entry in entries),
            retrieved_error_types=tuple(entry.error_type for entry in entries),
            source_db_ids=tuple(entry.source_db_id for entry in entries),
            source_stream_positions=tuple(
                entry.source_stream_position for entry in entries
            ),
            dense_scores=tuple(item.dense_score for item in ranked),
            bm25_scores=tuple(item.bm25_score for item in ranked),
            type_match_scores=tuple(item.type_match_score for item in ranked),
            final_scores=tuple(item.final_score for item in ranked),
            method_name=variant,
            legal_pool_size_by_polarity={
                pool.polarity: len(pool.legal) for pool in pools
            },
            typed_pool_size_by_polarity={
                pool.polarity: len(pool.typed) for pool in pools
            },
            fallback_used_by_polarity={
                pool.polarity: pool.fallback_used for pool in pools
            },
        )
        self.logs.append(log)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        log.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return RetrievalResult(
            positive_entries=tuple(
                entry for entry in entries if entry.polarity == "positive"
            ),
            negative_entries=tuple(
                entry for entry in entries if entry.polarity == "negative"
            ),
            log=log,
        )


__all__ = [
    "CONFIDENCE_AWARE_VARIANT",
    "CONDITIONALLY_TYPED_VARIANTS",
    "HybridRetriever",
    "MEMORYLESS_VARIANTS",
    "RetrievalLog",
    "RetrievalResult",
    "SUPPORTED_VARIANTS",
    "TYPED_VARIANTS",
    "UNTYPED_VARIANTS",
    "retrieval_log_expected_count",
    "validate_retrieval_log_alignment",
]
