"""Single causal runner for Spider, BIRD, baselines, and ablations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baselines import MethodPolicy, policy_for_config
from .classifier import ErrorClassification, classify_outcome
from .config import ExperimentConfig, canonical_json, content_hash, save_config
from .dataset import DatasetBundle, QueryExample
from .evaluator import Evaluator, OfficialEvaluatorAdapter
from .feedback import FeedbackRegime, Outcome, OutcomeStatus
from .generation import (
    GenerationBackend,
    InitialCacheIdentity,
    InitialSQLCache,
    InitialSQLCacheRecord,
    TransformersBackend,
    generate_reflection,
    generate_repair,
)
from .memory import MemoryEntry, MemoryStore
from .metrics import derive_metrics
from .prompts import (
    StaticExample,
    build_initial_prompt,
    build_reflection_prompt,
    build_repair_prompt,
)
from .retrieval import (
    HybridRetriever,
    RetrievalResult,
    retrieval_log_expected_count,
    validate_retrieval_log_alignment,
)


MANIFEST_FIELDS = (
    "run_id",
    "source_hash",
    "config_hash",
    "dataset_checksum",
    "database_manifest_hash",
    "model_name",
    "model_revision",
    "tokenizer_revision",
    "package_versions",
    "random_seed",
    "stream_order_seed",
    "feedback_regime",
    "evaluator_protocol",
    "evaluator_protocol_hash",
    "method_name",
    "initial_cache_hash",
)
_RESUME_IDENTITY_FIELDS = MANIFEST_FIELDS + ("static_examples_hash",)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    source_hash: str
    config_hash: str
    dataset_checksum: str
    database_manifest_hash: str
    model_name: str
    model_revision: str
    tokenizer_revision: str
    package_versions: Mapping[str, str]
    random_seed: int
    stream_order_seed: int
    feedback_regime: str
    evaluator_protocol: str
    evaluator_protocol_hash: str
    method_name: str
    initial_cache_hash: str
    static_examples_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResult:
    trajectory: Mapping[str, Any]
    prediction: Mapping[str, Any]
    memory_entries: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class RunResult:
    run_directory: Path
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    official_evaluation: Mapping[str, Any]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    line = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSONL") from error
    return rows


def _rewrite_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )
    os.replace(temporary, path)


def source_hash(source_root: str | Path | None = None) -> str:
    """Hash all live package/script source plus the pinned runtime specification."""

    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    paths = sorted((root / "merit").glob("*.py")) + sorted(
        (root / "scripts").glob("*.py")
    )
    pinned_runtime_paths = (
        "requirements.lock",
        "official_evaluation/evaluate_run.py",
        "official_evaluation/evaluate_cache.py",
        "official_evaluation/requirements.lock",
        "official_evaluation/vendor/bird/evaluation.py",
        "official_evaluation/vendor/bird/run_evaluation.sh",
        "official_evaluation/vendor/spider/evaluation.py",
        "official_evaluation/vendor/spider/exec_eval.py",
        "official_evaluation/vendor/spider/exec_subprocess.py",
        "official_evaluation/vendor/spider/parse.py",
        "official_evaluation/vendor/spider/process_sql.py",
    )
    paths.extend(
        path
        for relative in pinned_runtime_paths
        if (path := root / relative).is_file()
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_package_versions(
    expected: Mapping[str, str],
    *,
    enforce_pins: bool = True,
) -> dict[str, str]:
    if enforce_pins and not expected:
        raise RuntimeError("reportable runs require non-empty pinned package versions")
    versions: dict[str, str] = {}
    for package, pinned in sorted(expected.items()):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required pinned package is not installed: {package}") from error
        if enforce_pins and installed != pinned:
            raise RuntimeError(
                f"package revision mismatch for {package}: expected={pinned}, "
                f"installed={installed}"
            )
        versions[package] = installed
    return versions


def set_reproducible_seeds(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU/CUDA, and deterministic generation sources."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy

        numpy.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def _generation_seed(
    base_seed: int,
    query_id: str,
    stream_position: int,
    repair_iteration: int,
    purpose: str,
) -> int:
    payload = canonical_json(
        [base_seed, query_id, stream_position, repair_iteration, purpose]
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _episode_terminal(outcome: Outcome) -> bool:
    if outcome.feedback_regime == FeedbackRegime.DENOTATION_CONFIRMED.value:
        return outcome.status == OutcomeStatus.CORRECT
    return outcome.predicted_exec_ok


def _classification(outcome: Outcome) -> ErrorClassification | None:
    if outcome.status == OutcomeStatus.CORRECT:
        return None
    return classify_outcome(None, outcome)


def _failure_context(sql: str, outcome: Outcome) -> str:
    return (
        f"attempted_sql={sql.strip()}; status={outcome.status.value}; "
        f"db_error={outcome.db_error or '(none)'}"
    )


def _sql_delta(before: str | None, after: str) -> str:
    if before is None:
        return f"initial SQL => {after.strip()}"
    return f"{before.strip()} => {after.strip()}"


def _memory_has_episode(memory: MemoryStore, episode_id: str) -> bool:
    return any(
        str(observation.get("source_episode_id")) == episode_id
        for entry in memory.entries
        for observation in entry.provenance_history
    )


def _accounting(
    *,
    initial_prompt_tokens: int,
    initial_output_tokens: int,
    repair_prompt_tokens: int,
    repair_output_tokens: int,
    llm_calls: int,
    db_executions: int,
    embedding_calls: int,
    retrieval_calls: int,
) -> dict[str, int]:
    total_prompt = initial_prompt_tokens + repair_prompt_tokens
    total_output = initial_output_tokens + repair_output_tokens
    return {
        "initial_prompt_tokens": initial_prompt_tokens,
        "initial_output_tokens": initial_output_tokens,
        "repair_prompt_tokens": repair_prompt_tokens,
        "repair_output_tokens": repair_output_tokens,
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "total_tokens": total_prompt + total_output,
        "llm_calls": llm_calls,
        "db_executions": db_executions,
        "embedding_calls": embedding_calls,
        "retrieval_calls": retrieval_calls,
    }


def _load_static_examples(path: str) -> tuple[StaticExample, ...]:
    if not path:
        return ()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"static examples not found: {source}")
    if source.suffix.lower() == ".jsonl":
        rows = _read_jsonl(source)
    else:
        rows = _read_json(source)
        if not isinstance(rows, list):
            raise ValueError("static examples must be a JSON list or JSONL")
    examples = tuple(StaticExample(**dict(row)) for row in rows)
    for example in examples:
        example.validate()
        if example.source.strip().lower() in {"dev", "test", "evaluation", "current"}:
            raise ValueError("vanilla examples cannot come from the current evaluation split")
    return examples


class ExperimentRunner:
    """Run every method through one evaluator, cache, stream, and accounting path."""

    def __init__(
        self,
        config: ExperimentConfig,
        dataset: DatasetBundle,
        initial_cache: InitialSQLCache,
        evaluator: Evaluator,
        generation_backend: GenerationBackend,
        *,
        embedder: Any = None,
        vector_backend_factory: Any = None,
        enforce_package_pins: bool = True,
        source_root: str | Path | None = None,
    ):
        config.validate()
        if dataset.name != config.dataset.name:
            raise ValueError("dataset bundle does not match config")
        if dataset.invalid_examples:
            raise ValueError(
                f"dataset contains {len(dataset.invalid_examples)} invalid examples"
            )
        if not dataset.examples:
            raise ValueError("dataset contains no valid examples")
        if evaluator.dataset_config != config.dataset:
            raise ValueError("evaluator dataset config does not match run config")
        if evaluator.feedback_regime != config.feedback_regime:
            raise ValueError("evaluator feedback regime does not match run config")
        if evaluator.reference_timeout_seconds != float(
            config.reference_timeout_seconds
        ):
            raise ValueError(
                "evaluator reference timeout does not match run config"
            )
        if evaluator.evaluation_protocol != config.evaluation_protocol:
            raise ValueError(
                "evaluator protocol does not match run config"
            )
        if evaluator.evaluation_protocol_hash != config.evaluation_protocol_hash:
            raise ValueError(
                "evaluator protocol hash does not match run config"
            )
        self.config = config
        self.dataset = dataset
        self.initial_cache = initial_cache
        self.evaluator = evaluator
        self.generation_backend = generation_backend
        self.embedder = embedder
        self.vector_backend_factory = vector_backend_factory
        self.enforce_package_pins = enforce_package_pins
        self.source_root = (
            Path(source_root)
            if source_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.policy = policy_for_config(config)
        self.static_examples = (
            _load_static_examples(config.static_examples_path)
            if self.policy.static_example_source is not None
            else ()
        )
        expected_static_examples = (
            config.retrieval.max_positive + config.retrieval.max_negative
        )
        if (
            self.policy.static_example_source is not None
            and len(self.static_examples) != expected_static_examples
        ):
            raise ValueError("vanilla requires exactly K fixed static examples")
        self.static_examples_hash = (
            content_hash([asdict(example) for example in self.static_examples])
            if self.static_examples
            else ""
        )
        self._validate_backend()
        self._cache_by_query = self._validate_initial_cache()

    def _validate_backend(self) -> None:
        if isinstance(self.generation_backend, TransformersBackend):
            if self.generation_backend.model_config != self.config.model:
                raise ValueError("generation backend model identity differs from config")
            if self.generation_backend.generation_config != self.config.generation:
                raise ValueError("generation backend token limits differ from config")

    def _validate_initial_cache(self) -> dict[tuple[str, str], InitialSQLCacheRecord]:
        identity = InitialCacheIdentity.from_config(
            self.config,
            dataset_checksum=self.dataset.dataset_checksum,
            database_manifest_hash=self.dataset.database_manifest_hash,
        )
        prompt_hashes = {
            (example.query_id, example.db_id): build_initial_prompt(
                example.dataset_name,
                example.question,
                example.schema,
                example.evidence,
            ).hash
            for example in self.dataset
        }
        self.initial_cache.validate(
            expected_identity=identity,
            expected_prompt_hashes=prompt_hashes,
        )
        records = self.initial_cache.by_query()
        if len(records) != len(self.initial_cache.records):
            raise ValueError("initial cache query IDs are not canonically unique")
        if set(records) != set(prompt_hashes):
            missing = sorted(set(prompt_hashes) - set(records))
            extra = sorted(set(records) - set(prompt_hashes))
            raise ValueError(f"initial cache coverage mismatch; missing={missing}, extra={extra}")
        if any(record.oracle_correct is None for record in records.values()):
            raise ValueError("the shared main cache requires evaluator-confirmed oracle labels")
        return records

    def _manifest(self, run_id: str) -> RunManifest:
        return RunManifest(
            run_id=run_id,
            source_hash=source_hash(self.source_root),
            config_hash=self.config.hash,
            dataset_checksum=self.dataset.dataset_checksum,
            database_manifest_hash=self.dataset.database_manifest_hash,
            model_name=self.config.model.name,
            model_revision=self.config.model.revision,
            tokenizer_revision=self.config.model.tokenizer_revision,
            package_versions=installed_package_versions(
                self.config.package_versions,
                enforce_pins=self.enforce_package_pins,
            ),
            random_seed=self.config.random_seed,
            stream_order_seed=self.config.stream_order_seed,
            feedback_regime=self.config.feedback_regime,
            evaluator_protocol=self.config.evaluation_protocol,
            evaluator_protocol_hash=self.config.evaluation_protocol_hash,
            method_name=self.config.method_name,
            initial_cache_hash=self.initial_cache.cache_hash,
            static_examples_hash=self.static_examples_hash,
        )

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return (
            f"{self.config.dataset.name}-{self.config.method_name}-"
            f"s{self.config.stream_order_seed}-{timestamp}-{self.config.hash[:8]}"
        )

    def _prepare_fresh(self, run_id: str | None) -> tuple[Path, RunManifest]:
        resolved_id = run_id or self._new_run_id()
        if not _RUN_ID_RE.fullmatch(resolved_id):
            raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")
        directory = Path(self.config.run_root) / resolved_id
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        manifest = self._manifest(resolved_id)
        save_config(self.config, directory / "config.json")
        _atomic_json(directory / "manifest.json", manifest.to_dict())
        for name in (
            "predictions.jsonl",
            "trajectories.jsonl",
            "retrieval_log.jsonl",
            "stdout.log",
        ):
            (directory / name).touch(exist_ok=False)
        _atomic_json(directory / "metrics.json", {"status": "incomplete"})
        _atomic_json(
            directory / "official_eval.json",
            {"official": False, "status": "incomplete"},
        )
        return directory, manifest

    def _prepare_resume(self, directory: str | Path) -> tuple[Path, RunManifest]:
        run_directory = Path(directory)
        existing_raw = _read_json(run_directory / "manifest.json")
        expected = self._manifest(str(existing_raw["run_id"]))
        mismatches = {
            field: (existing_raw.get(field), expected.to_dict()[field])
            for field in _RESUME_IDENTITY_FIELDS
            if existing_raw.get(field) != expected.to_dict()[field]
        }
        if mismatches:
            raise ValueError(
                "resume rejected because immutable run identity changed: "
                + canonical_json(mismatches)
            )
        persisted_config = _read_json(run_directory / "config.json")
        if content_hash(persisted_config) != self.config.hash:
            raise ValueError("resume rejected because persisted config hash differs")
        for name in (
            "predictions.jsonl",
            "trajectories.jsonl",
            "retrieval_log.jsonl",
            "memory_positive.jsonl",
            "memory_negative.jsonl",
            "stdout.log",
        ):
            if not (run_directory / name).is_file():
                raise ValueError(f"resume artifact is missing: {name}")
        return run_directory, expected

    def _stream(self) -> list[QueryExample]:
        stream = list(self.dataset.examples)
        random.Random(self.config.stream_order_seed).shuffle(stream)
        return stream

    @staticmethod
    def _validate_prefix(
        trajectories: Sequence[Mapping[str, Any]],
        stream: Sequence[QueryExample],
    ) -> None:
        if len(trajectories) > len(stream):
            raise ValueError("resume trajectories exceed the dataset")
        for position, trajectory in enumerate(trajectories):
            expected = stream[position]
            if (
                str(trajectory.get("query_id")) != expected.query_id
                or str(trajectory.get("db_id")) != expected.db_id
                or int(trajectory.get("stream_position", -1)) != position
            ):
                raise ValueError("resume trajectories are not a valid stream prefix")

    @staticmethod
    def _validate_predictions(
        trajectories: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(predictions) != len(trajectories):
            raise ValueError("resume prediction/trajectory counts differ")
        identity_fields = ("query_id", "db_id", "source_index", "stream_position")
        for trajectory, prediction in zip(trajectories, predictions):
            if any(
                prediction.get(field) != trajectory.get(field)
                for field in identity_fields
            ):
                raise ValueError("resume prediction identity differs from trajectory")
            if prediction.get("predicted_sql") != trajectory.get("final_sql"):
                raise ValueError("resume prediction SQL differs from trajectory")

    def _clean_retrieval_log(
        self,
        path: Path,
        trajectories: Sequence[Mapping[str, Any]],
    ) -> None:
        rows = _read_jsonl(path)
        expected_count = retrieval_log_expected_count(
            trajectories,
            self.config.method_name,
        )
        committed_rows = rows[:expected_count]
        validate_retrieval_log_alignment(
            committed_rows,
            trajectories,
            self.config.method_name,
            type_match_weight=self.config.retrieval.type_match_weight,
        )

        trailing_rows = rows[expected_count:]
        processed_query_ids = {
            str(trajectory["query_id"]) for trajectory in trajectories
        }
        if trailing_rows and not self.policy.uses_global_memory:
            raise ValueError(
                "resume retrieval log is invalid for a memoryless method"
            )
        for row_number, row in enumerate(
            trailing_rows, start=expected_count + 1
        ):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"resume retrieval log row {row_number} must be an object"
                )
            query_id = str(row.get("query_id", "")).strip()
            repair_iteration = row.get("repair_iteration")
            if (
                not query_id
                or row.get("method_name") != self.config.method_name
                or isinstance(repair_iteration, bool)
                or not isinstance(repair_iteration, int)
                or repair_iteration < 1
            ):
                raise ValueError(
                    f"resume retrieval log row {row_number} is malformed"
                )
            if query_id in processed_query_ids:
                raise ValueError(
                    f"resume retrieval log row {row_number} is stale for "
                    "a committed query"
                )
        if trailing_rows:
            _rewrite_jsonl(path, committed_rows)

    def _create_memory(
        self,
        directory: Path,
    ) -> tuple[MemoryStore | None, HybridRetriever | None]:
        if not self.policy.uses_global_memory:
            paths = tuple(
                directory / name
                for name in ("memory_positive.jsonl", "memory_negative.jsonl")
            )
            for path in paths:
                if not path.exists():
                    path.touch()
            if any(path.stat().st_size for path in paths):
                raise ValueError("memoryless methods require empty memory artifacts")
            return None, None
        memory = MemoryStore(
            directory,
            self.config,
            embedder=self.embedder,
            backend_factory=self.vector_backend_factory,
        )
        retriever = HybridRetriever(
            memory,
            self.config,
            method_name=self.config.method_name,
            random_seed=self.config.random_seed,
            log_path=directory / "retrieval_log.jsonl",
        )
        return memory, retriever

    def _retrieve(
        self,
        retriever: HybridRetriever,
        *,
        example: QueryExample,
        stream_position: int,
        repair_iteration: int,
        error_type: str,
        current_sql: str,
        outcome: Outcome,
    ) -> RetrievalResult:
        result = retriever.retrieve(
            query_id=example.query_id,
            db_id=example.db_id,
            stream_position=stream_position,
            repair_iteration=repair_iteration,
            current_error_type=error_type,
            question=example.question,
            current_sql=current_sql,
            failure_context=_failure_context(current_sql, outcome),
        )
        if self.policy.causal_query_stream:
            for entry in result.entries:
                if entry.source_stream_position >= stream_position:
                    raise AssertionError("causal retrieval returned a future entry")
                if entry.source_query_id == example.query_id:
                    raise AssertionError("causal retrieval returned a current-query entry")
                if self.policy.cross_database_only and entry.source_db_id == example.db_id:
                    raise AssertionError("cross-database retrieval returned the current database")
        return result

    def _memory_candidates(
        self,
        *,
        example: QueryExample,
        stream_position: int,
        episode_id: str,
        attempts: Sequence[str],
        outcomes: Sequence[Outcome],
        classifications: Sequence[ErrorClassification | None],
        retrieved_entry_ids: Sequence[Sequence[str]],
    ) -> tuple[MemoryEntry, ...]:
        if not self.policy.updates_global_memory:
            return ()
        final_outcome = outcomes[-1]
        final_iteration = len(attempts) - 1
        provenance = {
            "method_name": self.config.method_name,
            "feedback_regime": self.config.feedback_regime,
            "retrieved_entry_ids": list(retrieved_entry_ids[-1]) if retrieved_entry_ids else [],
        }
        scope = "cross_database" if self.policy.cross_database_only else "global"

        if final_iteration > 0 and final_outcome.oracle_correct is True:
            prior_outcome = outcomes[-2]
            prior_classification = classifications[-2]
            error_type = (
                prior_classification.error_type
                if prior_classification is not None
                else "Result Mismatch"
            )
            delta = _sql_delta(attempts[-2], attempts[-1])
            return (
                MemoryEntry(
                    source_query_id=example.query_id,
                    source_db_id=example.db_id,
                    source_episode_id=episode_id,
                    source_stream_position=stream_position,
                    source_iteration=final_iteration,
                    polarity="positive",
                    provenance=provenance,
                    failure_context=_failure_context(attempts[-2], prior_outcome),
                    error_type=error_type,
                    attempted_sql_delta=delta,
                    observed_outcome=final_outcome.status.value,
                    observed_db_error=final_outcome.db_error,
                    applicability_scope=scope,
                    question=example.question,
                    schema=example.schema,
                    successful_direction=f"Observed successful transition: {delta}",
                ),
            )

        if final_outcome.oracle_correct is True:
            return ()

        unconfirmed_executable = (
            self.config.feedback_regime == FeedbackRegime.DBMS_ONLY.value
            and final_outcome.predicted_exec_ok
        )
        negative_disabled = self.config.method_name in {"positive_only", "dynamic_rag"}
        if unconfirmed_executable or negative_disabled:
            return ()
        final_classification = classifications[-1]
        error_type = (
            final_classification.error_type
            if final_classification is not None
            else "Result Mismatch"
        )
        before = attempts[-2] if len(attempts) > 1 else None
        return (
            MemoryEntry(
                source_query_id=example.query_id,
                source_db_id=example.db_id,
                source_episode_id=episode_id,
                source_stream_position=stream_position,
                source_iteration=final_iteration,
                polarity="negative",
                provenance=provenance,
                failure_context=_failure_context(attempts[-1], final_outcome),
                error_type=error_type,
                attempted_sql_delta=_sql_delta(before, attempts[-1]),
                observed_outcome=final_outcome.status.value,
                observed_db_error=final_outcome.db_error,
                applicability_scope=scope,
                question=example.question,
                schema=example.schema,
            ),
        )

    def _run_episode(
        self,
        *,
        example: QueryExample,
        stream_position: int,
        initial_record: InitialSQLCacheRecord,
        memory: MemoryStore | None,
        retriever: HybridRetriever | None,
        initial_outcome: Outcome | None = None,
        initial_db_executions: int | None = None,
        prepass_embedding_calls: int = 0,
        defer_memory: bool = False,
    ) -> EpisodeResult:
        episode_id = content_hash(
            [
                self.config.dataset.name,
                self.config.method_name,
                self.config.stream_order_seed,
                stream_position,
                example.query_id,
            ]
        )
        memory_calls_before = memory.embedding_calls if memory is not None else 0
        retrieval_calls_before = retriever.retrieval_calls if retriever is not None else 0

        if initial_outcome is None:
            initial_evaluation = self.evaluator.evaluate_counted(
                example,
                initial_record.initial_sql,
            )
            outcome = initial_evaluation.outcome
            db_executions = initial_evaluation.db_executions
        else:
            if (
                isinstance(initial_db_executions, bool)
                or not isinstance(initial_db_executions, int)
                or initial_db_executions < 0
            ):
                raise ValueError(
                    "precomputed initial outcome requires exact DB executions"
                )
            outcome = initial_outcome
            db_executions = initial_db_executions
        if (
            self.config.feedback_regime == FeedbackRegime.DENOTATION_CONFIRMED.value
            and outcome.oracle_correct != initial_record.oracle_correct
        ):
            raise AssertionError("shared-cache Success@1 changed during the run")

        attempts = [initial_record.initial_sql]
        outcomes = [outcome]
        classifications: list[ErrorClassification | None] = [_classification(outcome)]
        retrieved_entry_ids: list[list[str]] = []
        reflections: list[str] = []
        prompt_tokens = [initial_record.prompt_tokens]
        output_tokens = [initial_record.output_tokens]
        call_kinds = ["initial_cache"]
        repair_prompt_tokens = 0
        repair_output_tokens = 0
        llm_calls = 1

        def reflect_after_failure(repair_iteration: int) -> None:
            nonlocal repair_prompt_tokens, repair_output_tokens, llm_calls
            if self.policy.reflection is None or _episode_terminal(outcomes[-1]):
                return
            classification = classifications[-1]
            reflection_prompt = build_reflection_prompt(
                example.dataset_name,
                example.question,
                example.schema,
                attempts,
                outcomes,
                current_error_type=(
                    classification.error_type if classification is not None else "Unknown"
                ),
                evidence=example.evidence,
            )
            reflection = generate_reflection(
                self.generation_backend,
                reflection_prompt,
                self.config.generation,
                seed=_generation_seed(
                    self.config.random_seed,
                    example.query_id,
                    stream_position,
                    repair_iteration,
                    "reflection",
                ),
            )
            reflections.append(reflection.text)
            prompt_tokens.append(reflection.prompt_tokens)
            output_tokens.append(reflection.output_tokens)
            call_kinds.append("reflection")
            repair_prompt_tokens += reflection.prompt_tokens
            repair_output_tokens += reflection.output_tokens
            llm_calls += reflection.llm_calls

        reflect_after_failure(0)
        repair_steps = 0
        while (
            self.policy.performs_repairs
            and not _episode_terminal(outcomes[-1])
            and repair_steps < self.config.max_repair_steps
        ):
            repair_steps += 1
            current_classification = classifications[-1]
            error_type = (
                current_classification.error_type
                if current_classification is not None
                else "Unknown"
            )
            positive_entries: Sequence[MemoryEntry] = ()
            negative_entries: Sequence[MemoryEntry] = ()
            if self.policy.uses_global_memory:
                if retriever is None:
                    raise RuntimeError("global-memory method has no retriever")
                retrieved = self._retrieve(
                    retriever,
                    example=example,
                    stream_position=stream_position,
                    repair_iteration=repair_steps,
                    error_type=error_type,
                    current_sql=attempts[-1],
                    outcome=outcomes[-1],
                )
                positive_entries = retrieved.positive_entries
                negative_entries = retrieved.negative_entries
                retrieved_entry_ids.append(
                    [entry.entry_id for entry in retrieved.entries]
                )
            else:
                retrieved_entry_ids.append([])

            repair_prompt = build_repair_prompt(
                example.dataset_name,
                example.question,
                example.schema,
                attempts,
                status=outcomes[-1].status.value,
                current_error_type=error_type,
                db_error=outcomes[-1].db_error,
                evidence=example.evidence,
                positive_entries=positive_entries,
                negative_entries=negative_entries,
                static_examples=self.static_examples,
                reflections=tuple(reflections),
            )
            generated = generate_repair(
                self.generation_backend,
                repair_prompt,
                self.config.generation,
                seed=_generation_seed(
                    self.config.random_seed,
                    example.query_id,
                    stream_position,
                    repair_steps,
                    "repair",
                ),
            )
            attempts.append(generated.sql)
            for call in generated.calls:
                prompt_tokens.append(call.prompt_tokens)
                output_tokens.append(call.output_tokens)
                call_kinds.append("repair")
            repair_prompt_tokens += generated.prompt_tokens
            repair_output_tokens += generated.output_tokens
            llm_calls += generated.llm_calls

            evaluation = self.evaluator.evaluate_counted(example, generated.sql)
            outcome = evaluation.outcome
            outcomes.append(outcome)
            classifications.append(_classification(outcome))
            db_executions += evaluation.db_executions
            reflect_after_failure(repair_steps)

        memory_entries = self._memory_candidates(
            example=example,
            stream_position=stream_position,
            episode_id=episode_id,
            attempts=attempts,
            outcomes=outcomes,
            classifications=classifications,
            retrieved_entry_ids=retrieved_entry_ids,
        )

        final_outcome = outcomes[-1]
        if final_outcome.oracle_correct is None:
            offline_evaluation = self.evaluator.evaluate_offline_counted(
                example,
                attempts[-1],
            )
            final_correct = offline_evaluation.outcome.oracle_correct is True
            db_executions += offline_evaluation.db_executions
        else:
            final_correct = final_outcome.oracle_correct is True
        initial_correct = initial_record.oracle_correct is True
        initial_failure_type = (
            None
            if initial_correct
            else (
                classifications[0].error_type
                if classifications[0] is not None
                else "Result Mismatch"
            )
        )
        final_failure_type = (
            None
            if final_correct
            else (
                classifications[-1].error_type
                if classifications[-1] is not None
                else "Result Mismatch"
            )
        )
        current_failure_type = (
            classifications[-1].error_type
            if classifications[-1] is not None
            else None
        )
        retrieval_calls = (
            retriever.retrieval_calls - retrieval_calls_before
            if retriever is not None
            else 0
        )
        retrieval_embedding_calls = (
            memory.embedding_calls - memory_calls_before
            if memory is not None
            else 0
        )
        planned_insertions = len(memory_entries)
        accounting = _accounting(
            initial_prompt_tokens=initial_record.prompt_tokens,
            initial_output_tokens=initial_record.output_tokens,
            repair_prompt_tokens=repair_prompt_tokens,
            repair_output_tokens=repair_output_tokens,
            llm_calls=llm_calls,
            db_executions=db_executions,
            embedding_calls=(
                prepass_embedding_calls
                + retrieval_embedding_calls
                + planned_insertions
            ),
            retrieval_calls=retrieval_calls,
        )
        trajectory = {
            "query_id": example.query_id,
            "db_id": example.db_id,
            "source_index": example.source_index,
            "stream_position": stream_position,
            "episode_id": episode_id,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "repair_steps": len(attempts) - 1,
            "attempts": attempts,
            "outcomes": [item.to_dict() for item in outcomes],
            "classifications": [
                item.to_dict() if item is not None else None
                for item in classifications
            ],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "call_kinds": call_kinds,
            "retrieved_entry_ids": retrieved_entry_ids,
            "initial_failure_type": initial_failure_type,
            "current_failure_type": current_failure_type,
            "final_failure_type": final_failure_type,
            "accounting": accounting,
            "final_sql": attempts[-1],
            "static_example_sources": [item.source for item in self.static_examples],
            "static_examples_hash": self.static_examples_hash or None,
        }
        if defer_memory:
            trajectory["deferred_memory_entries"] = [
                entry.to_dict() for entry in memory_entries
            ]

        prediction = {
            "query_id": example.query_id,
            "db_id": example.db_id,
            "source_index": example.source_index,
            "stream_position": stream_position,
            "predicted_sql": attempts[-1],
        }
        return EpisodeResult(trajectory, prediction, memory_entries)

    @staticmethod
    def _commit_episode(
        directory: Path,
        result: EpisodeResult,
        memory: MemoryStore | None,
        *,
        defer_memory: bool = False,
    ) -> None:
        pending_path = directory / "episode_pending.json"
        _atomic_json(
            pending_path,
            {
                "trajectory": result.trajectory,
                "prediction": result.prediction,
                "memory_entries": [entry.to_dict() for entry in result.memory_entries],
                "defer_memory": defer_memory,
            },
        )
        if memory is not None and not defer_memory:
            for entry in result.memory_entries:
                if not _memory_has_episode(memory, entry.source_episode_id):
                    memory.add_entry(entry)
        _append_jsonl(directory / "trajectories.jsonl", result.trajectory)
        _append_jsonl(directory / "predictions.jsonl", result.prediction)
        pending_path.unlink()

    @staticmethod
    def _recover_pending(directory: Path, memory: MemoryStore | None) -> None:
        pending_path = directory / "episode_pending.json"
        if not pending_path.is_file():
            return
        pending = _read_json(pending_path)
        defer_memory = pending.get("defer_memory", False)
        if not isinstance(defer_memory, bool):
            raise ValueError("pending defer_memory must be boolean")
        pending_trajectory = pending["trajectory"]
        pending_prediction = pending["prediction"]
        query_id = str(pending_trajectory["query_id"])
        trajectories_path = directory / "trajectories.jsonl"
        predictions_path = directory / "predictions.jsonl"
        trajectory_matches = [
            row
            for row in _read_jsonl(trajectories_path)
            if str(row["query_id"]) == query_id
        ]
        prediction_matches = [
            row
            for row in _read_jsonl(predictions_path)
            if str(row["query_id"]) == query_id
        ]
        if len(trajectory_matches) > 1 or len(prediction_matches) > 1:
            raise ValueError("pending episode has duplicate persisted rows")
        if trajectory_matches and trajectory_matches[0] != pending_trajectory:
            raise ValueError("persisted trajectory conflicts with pending episode")
        if prediction_matches and prediction_matches[0] != pending_prediction:
            raise ValueError("persisted prediction conflicts with pending episode")
        if memory is not None and not defer_memory:
            for raw in pending["memory_entries"]:
                entry = MemoryEntry.from_dict(raw)
                if not _memory_has_episode(memory, entry.source_episode_id):
                    memory.add_entry(entry)
        if not trajectory_matches:
            _append_jsonl(trajectories_path, pending_trajectory)
        if not prediction_matches:
            _append_jsonl(predictions_path, pending_prediction)
        pending_path.unlink()

    @staticmethod
    def _commit_deferred_memory(
        trajectories: Sequence[Mapping[str, Any]],
        memory: MemoryStore,
    ) -> None:
        for trajectory in trajectories:
            raw_entries = trajectory.get("deferred_memory_entries")
            if not isinstance(raw_entries, list):
                raise ValueError("transductive trajectory is missing deferred memory")
            if len(raw_entries) > 1:
                raise ValueError(
                    "transductive episode produced more than one memory entry"
                )
            for raw in raw_entries:
                if not isinstance(raw, Mapping):
                    raise ValueError("deferred memory entry must be an object")
                entry = MemoryEntry.from_dict(raw)
                if (
                    entry.source_query_id != str(trajectory["query_id"])
                    or entry.source_db_id != str(trajectory["db_id"])
                    or entry.source_stream_position != trajectory["stream_position"]
                    or entry.source_episode_id != str(trajectory["episode_id"])
                ):
                    raise ValueError("deferred memory identity differs from trajectory")
                if not _memory_has_episode(memory, entry.source_episode_id):
                    memory.add_entry(entry)

    def _transductive_prepass_identity(
        self,
        stream: Sequence[QueryExample],
    ) -> dict[str, Any]:
        return {
            "config_hash": self.config.hash,
            "dataset_checksum": self.dataset.dataset_checksum,
            "database_manifest_hash": self.dataset.database_manifest_hash,
            "initial_cache_hash": self.initial_cache.cache_hash,
            "stream": [
                {
                    "query_id": example.query_id,
                    "db_id": example.db_id,
                    "stream_position": position,
                }
                for position, example in enumerate(stream)
            ],
        }

    def _validated_transductive_prepass(
        self,
        data: Any,
        stream: Sequence[QueryExample],
    ) -> dict[str, Mapping[str, Any]]:
        if not isinstance(data, Mapping):
            raise ValueError("transductive prepass must be a JSON object")
        if data.get("identity") != self._transductive_prepass_identity(stream):
            raise ValueError("transductive prepass identity differs from this run")
        rows = data.get("queries")
        if not isinstance(rows, list) or len(rows) != len(stream):
            raise ValueError("transductive prepass coverage differs from the stream")

        validated: dict[str, Mapping[str, Any]] = {}
        for position, (example, row) in enumerate(zip(stream, rows)):
            if not isinstance(row, Mapping):
                raise ValueError("transductive prepass contains a non-object row")
            if (
                row.get("query_id") != example.query_id
                or row.get("db_id") != example.db_id
                or row.get("stream_position") != position
            ):
                raise ValueError("transductive prepass coverage differs from the stream")
            outcome = Outcome.from_dict(row["outcome"])
            if outcome.feedback_regime != self.config.feedback_regime:
                raise ValueError("transductive prepass feedback regime differs")
            db_executions = row.get("db_executions")
            if (
                isinstance(db_executions, bool)
                or not isinstance(db_executions, int)
                or db_executions < 0
            ):
                raise ValueError("transductive prepass DB accounting is invalid")
            expected_embeddings = int(not _episode_terminal(outcome))
            if row.get("embedding_calls") != expected_embeddings:
                raise ValueError("transductive prepass embedding accounting differs")
            validated[example.query_id] = row
        return validated

    def _transductive_prepass(
        self,
        directory: Path,
        stream: Sequence[QueryExample],
        memory: MemoryStore,
    ) -> dict[str, Mapping[str, Any]]:
        path = directory / "transductive_prepass.json"
        if path.is_file():
            return self._validated_transductive_prepass(_read_json(path), stream)
        rows = []
        for position, example in enumerate(stream):
            record = self._cache_by_query[(example.query_id, example.db_id)]
            evaluation = self.evaluator.evaluate_counted(
                example,
                record.initial_sql,
            )
            outcome = evaluation.outcome
            classification = _classification(outcome)
            episode_id = content_hash(
                ["transductive_prepass", position, example.query_id]
            )
            insertion_count = 0
            online_failed = not _episode_terminal(outcome)
            if online_failed:
                error_type = (
                    classification.error_type
                    if classification is not None
                    else "Result Mismatch"
                )
                entry = MemoryEntry(
                    source_query_id=example.query_id,
                    source_db_id=example.db_id,
                    source_episode_id=episode_id,
                    source_stream_position=position,
                    source_iteration=0,
                    polarity="negative",
                    provenance={
                        "method_name": "transductive_batch",
                        "phase": "synchronous_initial_batch",
                    },
                    failure_context=_failure_context(record.initial_sql, outcome),
                    error_type=error_type,
                    attempted_sql_delta=_sql_delta(None, record.initial_sql),
                    observed_outcome=outcome.status.value,
                    observed_db_error=outcome.db_error,
                    applicability_scope="global",
                    question=example.question,
                    schema=example.schema,
                )
                if not _memory_has_episode(memory, episode_id):
                    memory.add_entry(entry)
                insertion_count = 1
            rows.append(
                {
                    "query_id": example.query_id,
                    "db_id": example.db_id,
                    "stream_position": position,
                    "outcome": outcome.to_dict(),
                    "db_executions": evaluation.db_executions,
                    "embedding_calls": insertion_count,
                }
            )
        payload = {
            "identity": self._transductive_prepass_identity(stream),
            "queries": rows,
        }
        _atomic_json(path, payload)
        return self._validated_transductive_prepass(payload, stream)

    def run(
        self,
        *,
        fresh_run: bool = False,
        resume_directory: str | Path | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        if fresh_run == (resume_directory is not None):
            raise ValueError("select exactly one of fresh_run or resume_directory")
        set_reproducible_seeds(self.config.random_seed)
        if fresh_run:
            directory, manifest = self._prepare_fresh(run_id)
        else:
            directory, manifest = self._prepare_resume(resume_directory or "")

        memory, retriever = self._create_memory(directory)
        self._recover_pending(directory, memory)
        stream = self._stream()
        trajectories = _read_jsonl(directory / "trajectories.jsonl")
        predictions = _read_jsonl(directory / "predictions.jsonl")
        self._validate_prefix(trajectories, stream)
        self._validate_predictions(trajectories, predictions)
        self._clean_retrieval_log(
            directory / "retrieval_log.jsonl",
            trajectories,
        )

        transductive = self.config.method_name == "transductive_batch"
        prepass: dict[str, Mapping[str, Any]] = {}
        if transductive:
            if memory is None:
                raise RuntimeError("transductive_batch requires memory")
            prepass = self._transductive_prepass(directory, stream, memory)

        for position in range(len(trajectories), len(stream)):
            example = stream[position]
            initial_record = self._cache_by_query[(example.query_id, example.db_id)]
            precomputed = None
            precomputed_db_executions = None
            prepass_embeddings = 0
            if transductive:
                row = prepass[example.query_id]
                precomputed = Outcome.from_dict(row["outcome"])
                precomputed_db_executions = int(row["db_executions"])
                prepass_embeddings = int(row["embedding_calls"])
            result = self._run_episode(
                example=example,
                stream_position=position,
                initial_record=initial_record,
                memory=memory,
                retriever=retriever,
                initial_outcome=precomputed,
                initial_db_executions=precomputed_db_executions,
                prepass_embedding_calls=prepass_embeddings,
                defer_memory=transductive,
            )
            self._commit_episode(directory, result, memory, defer_memory=transductive)
            message = (
                f"[{position + 1}/{len(stream)}] query={example.query_id} "
                f"initial={result.trajectory['initial_correct']} "
                f"final={result.trajectory['final_correct']} "
                f"repairs={result.trajectory['repair_steps']}"
            )
            with (directory / "stdout.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(message, flush=True)

        trajectories = _read_jsonl(directory / "trajectories.jsonl")
        if transductive:
            self._commit_deferred_memory(trajectories, memory)
        if memory is not None:
            memory.assert_consistent()
        predictions = _read_jsonl(directory / "predictions.jsonl")
        metrics = derive_metrics(trajectories)
        _atomic_json(directory / "metrics.json", metrics)
        if self.config.official_evaluator_command:
            official = OfficialEvaluatorAdapter(
                self.config.dataset,
                self.config.official_evaluator_command,
            ).evaluate(directory / "predictions.jsonl")
        else:
            official = {
                "official": False,
                "dataset_name": self.config.dataset.name,
                "final_correct_count": metrics["final_correct_count"],
                "reason": "official_evaluator_command is not configured",
            }
        _atomic_json(directory / "official_eval.json", official)
        manifest_payload = manifest.to_dict()
        manifest_payload["completed"] = True
        manifest_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(directory / "manifest.json", manifest_payload)
        return RunResult(directory, manifest_payload, metrics, official)


__all__ = [
    "MANIFEST_FIELDS",
    "EpisodeResult",
    "ExperimentRunner",
    "RunManifest",
    "RunResult",
    "installed_package_versions",
    "set_reproducible_seeds",
    "source_hash",
]
