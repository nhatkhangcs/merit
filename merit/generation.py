"""Gold-free generation, exact token accounting, and shared initial SQL caches."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .config import (
    DATASETS,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    canonical_json,
    content_hash,
)
from .prompts import PROMPT_FORMAT_VERSION, Prompt, build_initial_prompt


CACHE_FORMAT_VERSION = "merit-initial-cache-v5"
GREEDY_PROTOCOL = "greedy-one-sql-v1"
SELF_CONSISTENCY_PROTOCOL = "non-oracle-self-consistency-v1"
CONFIRMED_ANNOTATION_REGIME = "denotation_confirmed"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIRMED_STATUS_TO_ORACLE = {
    "CORRECT": True,
    "EXECUTION_ERROR": False,
    "DENOTATION_MISMATCH": False,
    "TIMEOUT": False,
}
_QUOTED_SQL_RE = re.compile(
    r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\])",
    re.DOTALL,
)
_SELECTION_TOKEN_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|"
    r"\[(?:\]\]|[^\]])*\]|\w+|[^\w\s]",
    re.DOTALL,
)


def _validate_token_count(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def _validate_positive_int(value: Any, name: str) -> None:
    _validate_token_count(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BackendGeneration:
    """One backend call measured from token IDs, never decoded/re-tokenized."""

    text: str
    prompt_tokens: int
    output_tokens: int

    def validate(self) -> None:
        _validate_token_count(self.prompt_tokens, "prompt_tokens")
        _validate_token_count(self.output_tokens, "output_tokens")


class GenerationBackend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        do_sample: bool,
        temperature: float,
        seed: int | None,
    ) -> BackendGeneration:
        """Generate one response and return token-ID-derived counts."""


@dataclass(frozen=True)
class GenerationCall:
    prompt_hash: str
    raw_response: str
    sql: str
    prompt_tokens: int
    output_tokens: int
    do_sample: bool
    seed: int | None

    def validate(self) -> None:
        if not isinstance(self.prompt_hash, str) or not _SHA256_RE.fullmatch(
            self.prompt_hash
        ):
            raise ValueError("generation call prompt_hash must be a SHA-256 digest")
        _validate_token_count(self.prompt_tokens, "prompt_tokens")
        _validate_token_count(self.output_tokens, "output_tokens")
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be boolean")


@dataclass(frozen=True)
class GenerationResult:
    """A selected SQL plus every LLM call made to select it."""

    sql: str
    calls: tuple[GenerationCall, ...]
    selected_index: int
    selection_method: str

    def validate(self) -> None:
        if not self.calls:
            raise ValueError("a generation result requires at least one call")
        if isinstance(self.selected_index, bool) or not isinstance(self.selected_index, int):
            raise TypeError("selected_index must be an integer")
        if not 0 <= self.selected_index < len(self.calls):
            raise ValueError("selected_index is outside the call list")
        if self.sql != self.calls[self.selected_index].sql:
            raise ValueError("selected SQL must match the selected call")
        for call in self.calls:
            call.validate()

    @property
    def prompt_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def llm_calls(self) -> int:
        return len(self.calls)


def _top_level_semicolons(sql: str) -> tuple[int, ...]:
    """Return semicolons outside SQL quotes and comments."""

    positions: list[int] = []
    state = "normal"
    index = 0
    while index < len(sql):
        character = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "normal":
            if character == "'":
                state = "single_quote"
            elif character == '"':
                state = "double_quote"
            elif character == "`":
                state = "backtick_quote"
            elif character == "[":
                state = "bracket_quote"
            elif character == "-" and following == "-":
                state = "line_comment"
                index += 1
            elif character == "/" and following == "*":
                state = "block_comment"
                index += 1
            elif character == ";":
                positions.append(index)
        elif state == "single_quote" and character == "'":
            if following == "'":
                index += 1
            else:
                state = "normal"
        elif state == "double_quote" and character == '"':
            if following == '"':
                index += 1
            else:
                state = "normal"
        elif state == "backtick_quote" and character == "`":
            if following == "`":
                index += 1
            else:
                state = "normal"
        elif state == "bracket_quote" and character == "]":
            state = "normal"
        elif state == "line_comment" and character in "\r\n":
            state = "normal"
        elif state == "block_comment" and character == "*" and following == "/":
            state = "normal"
            index += 1
        index += 1
    return tuple(positions)


def _is_sql_trivia(value: str) -> bool:
    """Return whether a SQL suffix contains only whitespace or comments."""

    index = 0
    while index < len(value):
        if value[index].isspace():
            index += 1
            continue
        if value.startswith("--", index):
            newline = value.find("\n", index + 2)
            if newline < 0:
                return True
            index = newline + 1
            continue
        if value.startswith("/*", index):
            closing = value.find("*/", index + 2)
            if closing < 0:
                return False
            index = closing + 2
            continue
        return False
    return True


def _single_sql_statement(sql: str) -> str:
    if not isinstance(sql, str):
        raise TypeError("generated SQL must be text")
    candidate = sql.strip()
    if not candidate:
        raise ValueError("generated SQL cannot be empty")
    semicolons = _top_level_semicolons(candidate)
    if len(semicolons) > 1:
        raise ValueError("generated response contains multiple SQL statements")
    if semicolons:
        delimiter = semicolons[0]
        if not _is_sql_trivia(candidate[delimiter + 1 :]):
            raise ValueError("generated response contains multiple SQL statements")
        candidate = candidate[:delimiter].rstrip()
    if not candidate:
        raise ValueError("generated SQL cannot be empty")
    return candidate


def _single_sql_or_empty(sql: str) -> str:
    """Canonicalize one SQL statement or return an invalid empty prediction."""

    if isinstance(sql, str) and not sql.strip():
        return ""
    try:
        return _single_sql_statement(sql)
    except ValueError:
        return ""


def extract_sql(response: str) -> str:
    """Extract one SQL statement from the supported response wrappers."""

    answer = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    content = answer.group(1) if answer else response
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", content, re.IGNORECASE | re.DOTALL)
    content = fenced.group(1) if fenced else content
    start = re.search(r"\b(?:WITH|SELECT)\b", content, re.IGNORECASE)
    sql = content[start.start() :] if start else content
    sql = re.split(
        r"</?(?:answer|reflection)>|(?:^|\n)\s*(?:Human|Assistant)\s*:",
        sql,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _single_sql_or_empty(sql)


def normalize_sql_for_selection(sql: str) -> str:
    """Normalize presentation only; do not remove literals or inspect references."""

    parts = _QUOTED_SQL_RE.split(sql.strip().rstrip(";"))
    normalized: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            normalized.append(part)
            continue
        unquoted = re.sub(r"\s+", " ", part.casefold())
        normalized.append(
            re.sub(r"\s*([(),=<>+\-*/])\s*", r"\1", unquoted)
        )
    return "".join(normalized).strip()


def _selection_tokens(sql: str) -> frozenset[str]:
    return frozenset(_SELECTION_TOKEN_RE.findall(sql))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _select_non_oracle(candidates: Sequence[str]) -> tuple[int, str]:
    normalized = [normalize_sql_for_selection(sql) for sql in candidates]
    groups: dict[str, list[int]] = {}
    for index, sql in enumerate(normalized):
        groups.setdefault(sql, []).append(index)
    largest = max(len(indices) for indices in groups.values())
    largest_groups = [indices for indices in groups.values() if len(indices) == largest]
    if len(largest_groups) == 1:
        return largest_groups[0][0], "normalized_majority"

    token_sets = [_selection_tokens(sql) for sql in normalized]
    centrality = [
        sum(
            _jaccard(token_sets[index], token_sets[other])
            for other in range(len(token_sets))
            if other != index
        )
        / max(1, len(token_sets) - 1)
        for index in range(len(token_sets))
    ]
    best = max(range(len(candidates)), key=lambda index: (centrality[index], -index))
    return best, "normalized_centrality"


def _call_backend(
    backend: GenerationBackend,
    prompt: Prompt,
    *,
    max_output_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> GenerationCall:
    generated = backend.generate(
        prompt.text,
        max_output_tokens=max_output_tokens,
        do_sample=do_sample,
        temperature=temperature,
        seed=seed,
    )
    generated.validate()
    return GenerationCall(
        prompt_hash=prompt.hash,
        raw_response=generated.text,
        sql=extract_sql(generated.text),
        prompt_tokens=generated.prompt_tokens,
        output_tokens=generated.output_tokens,
        do_sample=do_sample,
        seed=seed,
    )


def generate_greedy_initial(
    backend: GenerationBackend,
    prompt: Prompt,
    config: GenerationConfig,
    *,
    seed: int | None = None,
) -> GenerationResult:
    """Generate exactly one initial SQL with the main greedy protocol."""

    config.validate()
    if prompt.purpose != "initial":
        raise ValueError("the shared initial cache requires an initial prompt")
    call = _call_backend(
        backend,
        prompt,
        max_output_tokens=config.max_output_tokens,
        do_sample=False,
        temperature=0.0,
        seed=seed,
    )
    result = GenerationResult(call.sql, (call,), 0, GREEDY_PROTOCOL)
    result.validate()
    return result


def generate_repair(
    backend: GenerationBackend,
    prompt: Prompt,
    config: GenerationConfig,
    *,
    seed: int | None = None,
) -> GenerationResult:
    config.validate()
    if prompt.purpose != "repair":
        raise ValueError("repair generation requires a repair prompt")
    do_sample = config.repair_temperature > 0
    if do_sample and seed is None:
        raise ValueError("sampled repair generation requires an explicit seed")
    call = _call_backend(
        backend,
        prompt,
        max_output_tokens=config.max_output_tokens,
        do_sample=do_sample,
        temperature=config.repair_temperature if do_sample else 0.0,
        seed=seed,
    )
    result = GenerationResult(call.sql, (call,), 0, "repair-generation-v1")
    result.validate()
    return result


@dataclass(frozen=True)
class ReflectionResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    llm_calls: int = 1

    def validate(self) -> None:
        _validate_token_count(self.prompt_tokens, "prompt_tokens")
        _validate_token_count(self.output_tokens, "output_tokens")
        _validate_positive_int(self.llm_calls, "llm_calls")


def generate_reflection(
    backend: GenerationBackend,
    prompt: Prompt,
    config: GenerationConfig,
    *,
    seed: int | None = None,
) -> ReflectionResult:
    """Execute the one-call Reflexion step; callers must account for this call."""

    config.validate()
    if prompt.purpose != "reflection":
        raise ValueError("reflection generation requires a reflection prompt")
    generated = backend.generate(
        prompt.text,
        max_output_tokens=config.max_output_tokens,
        do_sample=False,
        temperature=0.0,
        seed=seed,
    )
    generated.validate()
    match = re.search(
        r"<reflection>(.*?)</reflection>", generated.text, re.IGNORECASE | re.DOTALL
    )
    text = (
        match.group(1)
        if match
        else re.split(r"</reflection>", generated.text, maxsplit=1, flags=re.IGNORECASE)[0]
    ).strip()
    result = ReflectionResult(text, generated.prompt_tokens, generated.output_tokens)
    result.validate()
    return result


def self_consistency_seeds(base_seed: int, sample_count: int) -> tuple[int, ...]:
    if sample_count < 2:
        raise ValueError("self-consistency requires at least two samples")
    return tuple(base_seed + index for index in range(sample_count))


def generate_non_oracle_self_consistency(
    backend: GenerationBackend,
    prompt: Prompt,
    config: GenerationConfig,
    *,
    sample_count: int,
    base_seed: int,
    temperature: float = 0.4,
) -> GenerationResult:
    """Optional ablation selected only by normalized majority/centrality."""

    config.validate()
    if prompt.purpose != "initial":
        raise ValueError("self-consistency requires an initial prompt")
    if temperature <= 0:
        raise ValueError("self-consistency sampling temperature must be positive")
    calls = tuple(
        _call_backend(
            backend,
            prompt,
            max_output_tokens=config.max_output_tokens,
            do_sample=True,
            temperature=temperature,
            seed=seed,
        )
        for seed in self_consistency_seeds(base_seed, sample_count)
    )
    selected_index, selection = _select_non_oracle([call.sql for call in calls])
    result = GenerationResult(
        calls[selected_index].sql,
        calls,
        selected_index,
        f"{SELF_CONSISTENCY_PROTOCOL}:{selection}",
    )
    result.validate()
    return result


@dataclass(frozen=True)
class InitialCacheIdentity:
    dataset_name: str
    dataset_checksum: str
    database_manifest_hash: str
    model_name: str
    model_revision: str
    tokenizer_revision: str
    quantization: str
    prompt_format_version: str
    max_input_tokens: int
    max_output_tokens: int
    evaluation_protocol: str
    evaluation_protocol_hash: str
    decoding_protocol: str = GREEDY_PROTOCOL
    annotation_regime: str = CONFIRMED_ANNOTATION_REGIME

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        dataset_checksum: str,
        database_manifest_hash: str,
    ) -> "InitialCacheIdentity":
        return cls(
            dataset_name=config.dataset.name,
            dataset_checksum=dataset_checksum,
            database_manifest_hash=database_manifest_hash,
            model_name=config.model.name,
            model_revision=config.model.revision,
            tokenizer_revision=config.model.tokenizer_revision,
            quantization=config.model.quantization,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            max_input_tokens=config.generation.max_input_tokens,
            max_output_tokens=config.generation.max_output_tokens,
            evaluation_protocol=config.evaluation_protocol,
            evaluation_protocol_hash=config.evaluation_protocol_hash,
        )

    def validate(self) -> None:
        if self.dataset_name not in DATASETS:
            raise ValueError(f"unsupported dataset: {self.dataset_name}")
        if not isinstance(self.dataset_checksum, str) or not _SHA256_RE.fullmatch(
            self.dataset_checksum
        ):
            raise ValueError("dataset_checksum must be a SHA-256 digest")
        if not isinstance(
            self.database_manifest_hash, str
        ) or not _SHA256_RE.fullmatch(self.database_manifest_hash):
            raise ValueError("database_manifest_hash must be a SHA-256 digest")
        if self.quantization not in {"4bit", "none"}:
            raise ValueError("cache quantization must be 4bit or none")
        ModelConfig(
            name=self.model_name,
            revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
        ).validate()
        if self.prompt_format_version != PROMPT_FORMAT_VERSION:
            raise ValueError("initial cache prompt format version is stale")
        if self.decoding_protocol != GREEDY_PROTOCOL:
            raise ValueError("the shared initial cache must use greedy one-SQL decoding")
        if self.annotation_regime != CONFIRMED_ANNOTATION_REGIME:
            raise ValueError("initial cache annotations must be denotation-confirmed")
        if (
            not isinstance(self.evaluation_protocol, str)
            or not self.evaluation_protocol
            or self.evaluation_protocol != self.evaluation_protocol.strip()
        ):
            raise ValueError("evaluation_protocol must be a non-empty canonical string")
        if not isinstance(
            self.evaluation_protocol_hash, str
        ) or not _SHA256_RE.fullmatch(self.evaluation_protocol_hash):
            raise ValueError("evaluation_protocol_hash must be a SHA-256 digest")
        _validate_positive_int(self.max_input_tokens, "max_input_tokens")
        _validate_positive_int(self.max_output_tokens, "max_output_tokens")

    @property
    def hash(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class InitialEvaluation:
    """Offline, denotation-confirmed annotation for an initial prediction."""

    execution_status: str
    oracle_correct: bool

    def __post_init__(self) -> None:
        status = getattr(self.execution_status, "value", self.execution_status)
        object.__setattr__(self, "execution_status", str(status))

    def validate(self) -> None:
        if not isinstance(self.oracle_correct, bool):
            raise TypeError("confirmed oracle_correct must be boolean")
        try:
            expected = _CONFIRMED_STATUS_TO_ORACLE[self.execution_status]
        except KeyError as exc:
            raise ValueError(
                f"unsupported confirmed execution status: {self.execution_status}"
            ) from exc
        if self.oracle_correct is not expected:
            raise ValueError(
                "execution_status and oracle_correct are inconsistent for "
                "denotation-confirmed annotation"
            )


@dataclass(frozen=True)
class InitialSQLCacheRecord:
    query_id: str
    db_id: str
    prompt_hash: str
    initial_sql: str
    prompt_tokens: int
    output_tokens: int
    execution_status: str
    oracle_correct: bool

    def validate(self) -> None:
        _validate_query_key(self.query_id, self.db_id)
        if not isinstance(self.prompt_hash, str) or not _SHA256_RE.fullmatch(
            self.prompt_hash
        ):
            raise ValueError("prompt_hash must be a SHA-256 digest")
        if _single_sql_or_empty(self.initial_sql) != self.initial_sql:
            raise ValueError(
                "initial_sql must be empty or one canonical SQL statement"
            )
        _validate_token_count(self.prompt_tokens, "prompt_tokens")
        _validate_token_count(self.output_tokens, "output_tokens")
        InitialEvaluation(self.execution_status, self.oracle_correct).validate()


@dataclass(frozen=True)
class InitialCacheQuery:
    query_id: str
    db_id: str
    question: str
    schema: str
    evidence: str = ""

    def validate(self) -> None:
        _validate_query_key(self.query_id, self.db_id)
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("initial cache question is required")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("initial cache schema is required")


def _validate_query_key(query_id: Any, db_id: Any) -> None:
    if not isinstance(query_id, str):
        raise TypeError("query_id must be a canonical string")
    if not query_id or query_id != query_id.strip():
        raise ValueError("query_id must be a non-empty canonical string")
    if not isinstance(db_id, str):
        raise TypeError("db_id must be a canonical string")
    if not db_id or db_id != db_id.strip():
        raise ValueError("db_id must be a non-empty canonical string")


def _record_key(record: InitialSQLCacheRecord) -> str:
    return canonical_json([record.query_id, record.db_id])


def _canonical_records(
    records: Iterable[InitialSQLCacheRecord],
) -> tuple[InitialSQLCacheRecord, ...]:
    return tuple(sorted(records, key=_record_key))


@dataclass(frozen=True)
class InitialSQLCache:
    identity: InitialCacheIdentity
    records: tuple[InitialSQLCacheRecord, ...]
    cache_hash: str
    format_version: str = CACHE_FORMAT_VERSION

    @classmethod
    def create(
        cls,
        identity: InitialCacheIdentity,
        records: Iterable[InitialSQLCacheRecord],
    ) -> "InitialSQLCache":
        """Create a cache from records already attested by the generation factory.

        Production cache construction goes through :func:`build_initial_cache` and
        :func:`cache_record_from_generation`; the persisted record schema purposely
        contains only the protocol-required fields and not generation internals.
        """

        canonical_records = _canonical_records(records)
        payload = _cache_payload(identity, canonical_records)
        cache = cls(identity, canonical_records, content_hash(payload))
        cache.validate()
        return cache

    def validate(
        self,
        *,
        expected_identity: InitialCacheIdentity | None = None,
        expected_prompt_hashes: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        if self.format_version != CACHE_FORMAT_VERSION:
            raise ValueError(f"unsupported initial cache format: {self.format_version}")
        self.identity.validate()
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError("initial cache identity does not match this run")
        expected_keys: set[tuple[str, str]] | None = None
        if expected_prompt_hashes is not None:
            expected_keys = set()
            for key, digest in expected_prompt_hashes.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise TypeError("expected prompt keys must be (query_id, db_id) tuples")
                _validate_query_key(key[0], key[1])
                if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                    raise ValueError("expected prompt hashes must be SHA-256 digests")
                expected_keys.add(key)

        keys: set[tuple[str, str]] = set()
        for record in self.records:
            record.validate()
            key = (record.query_id, record.db_id)
            if key in keys:
                raise ValueError(
                    f"duplicate initial cache record: {canonical_json(key)}"
                )
            keys.add(key)
            if expected_prompt_hashes is not None:
                if expected_prompt_hashes.get(key) != record.prompt_hash:
                    raise ValueError(f"prompt hash mismatch for query {record.query_id}")
        if expected_keys is not None and keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(
                f"initial cache coverage mismatch; missing={missing}, extra={extra}"
            )
        expected_hash = content_hash(_cache_payload(self.identity, self.records))
        if self.cache_hash != expected_hash:
            raise ValueError("initial cache content hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "identity": asdict(self.identity),
            "records": [asdict(record) for record in self.records],
            "cache_hash": self.cache_hash,
        }

    def by_query(self) -> dict[tuple[str, str], InitialSQLCacheRecord]:
        self.validate()
        records: dict[tuple[str, str], InitialSQLCacheRecord] = {}
        for record in self.records:
            key = (record.query_id, record.db_id)
            if key in records:
                raise ValueError(f"duplicate initial cache record: {canonical_json(key)}")
            records[key] = record
        return records


def _cache_payload(
    identity: InitialCacheIdentity,
    records: Iterable[InitialSQLCacheRecord],
) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "identity": asdict(identity),
        "records": [asdict(record) for record in _canonical_records(records)],
    }


def cache_record_from_generation(
    *,
    query_id: str,
    db_id: str,
    prompt: Prompt,
    generation: GenerationResult,
    evaluation: InitialEvaluation,
) -> InitialSQLCacheRecord:
    """Create a main-protocol record; sampled selection is rejected."""

    generation.validate()
    evaluation.validate()
    if prompt.purpose != "initial":
        raise ValueError("shared initial cache records require an initial prompt")
    if generation.selection_method != GREEDY_PROTOCOL or generation.llm_calls != 1:
        raise ValueError("shared initial cache records require exactly one greedy LLM call")
    call = generation.calls[0]
    if call.do_sample:
        raise ValueError("shared initial cache records require do_sample=false")
    if call.prompt_hash != prompt.hash:
        raise ValueError("generation prompt does not match the cache prompt")
    initial_sql = _single_sql_or_empty(generation.sql)
    record = InitialSQLCacheRecord(
        query_id=query_id,
        db_id=db_id,
        prompt_hash=prompt.hash,
        initial_sql=initial_sql,
        prompt_tokens=generation.prompt_tokens,
        output_tokens=generation.output_tokens,
        execution_status=evaluation.execution_status,
        oracle_correct=evaluation.oracle_correct,
    )
    record.validate()
    return record


InitialEvaluator = Callable[[str, str, str], InitialEvaluation | Any]


def _public_evaluation(value: InitialEvaluation | Any) -> InitialEvaluation:
    if isinstance(value, InitialEvaluation):
        value.validate()
        return value
    status = getattr(value, "status")
    status_value = getattr(status, "value", status)
    regime = getattr(value, "feedback_regime", None)
    regime_value = getattr(regime, "value", regime)
    if regime_value != CONFIRMED_ANNOTATION_REGIME:
        raise ValueError("initial cache evaluator must return denotation-confirmed output")
    evaluation = InitialEvaluation(
        str(status_value),
        getattr(value, "oracle_correct"),
    )
    evaluation.validate()
    return evaluation


def _attest_cache_backend(
    backend: GenerationBackend,
    config: ExperimentConfig,
) -> None:
    if isinstance(backend, TransformersBackend):
        if backend.model_config != config.model:
            raise ValueError("cache backend model identity differs from config")
        if backend.generation_config != config.generation:
            raise ValueError("cache backend generation identity differs from config")


def build_initial_cache(
    *,
    queries: Iterable[InitialCacheQuery],
    config: ExperimentConfig,
    dataset_checksum: str,
    database_manifest_hash: str,
    backend: GenerationBackend,
    evaluate: InitialEvaluator,
) -> InitialSQLCache:
    """Generate one greedy SQL per query and add offline confirmed annotations."""

    config.validate()
    _attest_cache_backend(backend, config)
    identity = InitialCacheIdentity.from_config(
        config,
        dataset_checksum=dataset_checksum,
        database_manifest_hash=database_manifest_hash,
    )
    identity.validate()
    query_list = tuple(
        InitialCacheQuery(
            query_id=query.query_id,
            db_id=query.db_id,
            question=query.question,
            schema=query.schema,
            evidence=getattr(query, "evidence", ""),
        )
        for query in queries
    )
    seen_query_keys: set[tuple[str, str]] = set()
    for query in query_list:
        query.validate()
        key = (query.query_id, query.db_id)
        if key in seen_query_keys:
            raise ValueError(f"duplicate initial cache query: {canonical_json(key)}")
        seen_query_keys.add(key)

    records: list[InitialSQLCacheRecord] = []
    expected_prompt_hashes: dict[tuple[str, str], str] = {}
    for query in query_list:
        prompt = build_initial_prompt(
            config.dataset.name,
            query.question,
            query.schema,
            query.evidence,
        )
        expected_prompt_hashes[(query.query_id, query.db_id)] = prompt.hash
        generation = generate_greedy_initial(
            backend,
            prompt,
            config.generation,
            seed=config.random_seed,
        )
        evaluation = _public_evaluation(
            evaluate(query.query_id, query.db_id, generation.sql)
        )
        records.append(
            cache_record_from_generation(
                query_id=query.query_id,
                db_id=query.db_id,
                prompt=prompt,
                generation=generation,
                evaluation=evaluation,
            )
        )
    cache = InitialSQLCache.create(identity, records)
    cache.validate(
        expected_identity=identity,
        expected_prompt_hashes=expected_prompt_hashes,
    )
    return cache


def save_initial_cache(cache: InitialSQLCache, path: str | Path) -> None:
    cache.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(cache.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def load_initial_cache(
    path: str | Path,
    *,
    expected_identity: InitialCacheIdentity | None = None,
    expected_prompt_hashes: Mapping[tuple[str, str], str] | None = None,
) -> InitialSQLCache:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported initial cache format: {raw.get('format_version')}"
        )
    cache = InitialSQLCache(
        identity=InitialCacheIdentity(**raw["identity"]),
        records=tuple(InitialSQLCacheRecord(**record) for record in raw["records"]),
        cache_hash=raw["cache_hash"],
        format_version=raw["format_version"],
    )
    cache.validate(
        expected_identity=expected_identity,
        expected_prompt_hashes=expected_prompt_hashes,
    )
    return cache


class TransformersBackend:
    """Pinned Hugging Face backend whose optional dependencies load on first use."""

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        model_config.validate()
        generation_config.validate()
        if model_config.quantization not in {"4bit", "none"}:
            raise ValueError("TransformersBackend supports quantization='4bit' or 'none'")
        self.model_config = model_config
        self.generation_config = generation_config
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        self.load()
        return self._model

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name,
            revision=self.model_config.tokenizer_revision,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
        )
        model_kwargs: dict[str, Any] = {
            "revision": self.model_config.revision,
            "device_map": self.model_config.device_map,
            "trust_remote_code": True,
        }
        if self.model_config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            **model_kwargs,
        )
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        do_sample: bool,
        temperature: float,
        seed: int | None,
    ) -> BackendGeneration:
        _validate_positive_int(max_output_tokens, "max_output_tokens")
        self.load()
        if do_sample and seed is None:
            raise ValueError("sampled generation requires an explicit seed")
        if do_sample and temperature <= 0:
            raise ValueError("sampled generation requires a positive temperature")

        rendered_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(
            [rendered_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.generation_config.max_input_tokens,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self._model.device)
        attention_mask = encoded["attention_mask"].to(self._model.device)
        prompt_tokens = int(attention_mask.sum().item())
        input_width = int(input_ids.shape[1])
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_output_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            generator = self._torch.Generator(device=input_ids.device)
            generator.manual_seed(seed)
            generation_kwargs["temperature"] = temperature
            generation_kwargs["generator"] = generator
        else:
            generation_kwargs["temperature"] = None
            generation_kwargs["top_p"] = None
            generation_kwargs["top_k"] = None

        with self._torch.no_grad():
            generated = self._model.generate(**generation_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        suffix = sequences[0, input_width:]
        output_tokens = int(suffix.shape[-1])
        text = self._tokenizer.decode(suffix, skip_special_tokens=True)
        return BackendGeneration(text, prompt_tokens, output_tokens)


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CONFIRMED_ANNOTATION_REGIME",
    "GREEDY_PROTOCOL",
    "SELF_CONSISTENCY_PROTOCOL",
    "BackendGeneration",
    "GenerationBackend",
    "GenerationCall",
    "GenerationResult",
    "InitialCacheIdentity",
    "InitialCacheQuery",
    "InitialEvaluation",
    "InitialSQLCache",
    "InitialSQLCacheRecord",
    "ReflectionResult",
    "TransformersBackend",
    "build_initial_cache",
    "cache_record_from_generation",
    "extract_sql",
    "generate_greedy_initial",
    "generate_non_oracle_self_consistency",
    "generate_reflection",
    "generate_repair",
    "load_initial_cache",
    "normalize_sql_for_selection",
    "save_initial_cache",
    "self_consistency_seeds",
]
