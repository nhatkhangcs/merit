"""Reference-isolated execution and official evaluation adapters.

No other MERIT module should load reference SQL or reference denotations.
Online callers receive only :class:`Outcome`; ``dbms_only`` never consults the
reference store.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    DatasetConfig,
    EVALUATION_PROTOCOLS,
    EVALUATION_TIMEOUT_SECONDS,
    FEEDBACK_REGIMES,
    evaluation_protocol_hash,
    evaluation_protocol_identity,
)
from .dataset import QueryExample, stable_query_id
from .feedback import FeedbackRegime, Outcome, OutcomeStatus


class EvaluationError(RuntimeError):
    """Raised when benchmark references or evaluator infrastructure are invalid."""


@dataclass(frozen=True)
class CountedEvaluation:
    """One public outcome plus the exact number of attempted SQL executions."""

    outcome: Outcome
    db_executions: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.db_executions, bool)
            or not isinstance(self.db_executions, int)
            or self.db_executions < 0
        ):
            raise ValueError("db_executions must be a non-negative integer")


@dataclass(frozen=True)
class ExecutionResult:
    """Internal result of one read-only SQLite execution."""

    rows: list[list[Any]] | None
    error: str | None
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return self.rows is not None and self.error is None and not self.timed_out


def _public_cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def _validated_timeout_seconds(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


_DENIED_SQLITE_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_UPDATE,
    }
)


def _read_only_authorizer(
    action_code: int,
    first_argument: str | None,
    second_argument: str | None,
    database_name: str | None,
    trigger_name: str | None,
) -> int:
    del first_argument, second_argument, database_name, trigger_name
    return (
        sqlite3.SQLITE_DENY
        if action_code in _DENIED_SQLITE_ACTIONS
        else sqlite3.SQLITE_OK
    )


def execute_read_only(
    database_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 15.0,
) -> ExecutionResult:
    """Execute one query through a read-only SQLite URI with a hard progress deadline."""

    timeout_seconds = _validated_timeout_seconds(
        timeout_seconds,
        "timeout_seconds",
    )
    query = str(sql).strip()
    path = Path(database_path).resolve()
    if not path.is_file():
        return ExecutionResult(None, f"database not found: {path}", False)

    deadline = time.monotonic() + timeout_seconds
    connection: sqlite3.Connection | None = None
    try:
        uri = f"{path.as_uri()}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=min(timeout_seconds, 5.0),
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_read_only_authorizer)
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() >= deadline else 0,
            500,
        )
        cursor = connection.execute(query)
        rows = [
            [_public_cell(value) for value in row]
            for row in cursor.fetchall()
        ]
        return ExecutionResult(rows, None, False)
    except sqlite3.Error as exc:
        message = str(exc)
        timed_out = (
            time.monotonic() >= deadline
            or "interrupted" in message.lower()
        )
        return ExecutionResult(
            None,
            "TIMEOUT" if timed_out else message,
            timed_out,
        )
    finally:
        if connection is not None:
            connection.close()


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL_ROOT = _REPOSITORY_ROOT / "official_evaluation"
_VENDOR_ROOT = _OFFICIAL_ROOT / "vendor"
_VENDOR_PYTHON = _VENDOR_ROOT / "python"
_SPIDER_MODULE_LOCK = threading.Lock()
_SPIDER_EXEC_MODULE: Any | None = None


@dataclass(frozen=True)
class _RawExecution:
    rows: list[tuple[Any, ...]] | None
    error: str | None
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return self.rows is not None and self.error is None and not self.timed_out


def _ensure_official_import_paths() -> None:
    for path in (_VENDOR_PYTHON, _VENDOR_ROOT / "spider"):
        rendered = str(path)
        if path.is_dir() and rendered not in sys.path:
            sys.path.insert(0, rendered)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_protocol_sources(dataset_name: str, identity: Mapping[str, Any]) -> None:
    source_hashes = identity.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise EvaluationError("evaluation protocol has no pinned source hashes")
    source_root = _VENDOR_ROOT / dataset_name
    for relative, expected in sorted(source_hashes.items()):
        path = source_root / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise EvaluationError(f"pinned evaluator source mismatch: {path}")


def _load_spider_exec_module() -> Any:
    global _SPIDER_EXEC_MODULE
    if _SPIDER_EXEC_MODULE is not None:
        return _SPIDER_EXEC_MODULE
    _ensure_official_import_paths()
    path = (_VENDOR_ROOT / "spider" / "exec_eval.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "merit_pinned_spider_exec_eval",
        path,
    )
    if spec is None or spec.loader is None:
        raise EvaluationError(f"cannot import pinned Spider evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise EvaluationError("imported Spider evaluator path differs from the pin")
    _SPIDER_EXEC_MODULE = module
    return module


def render_spider_prediction(sql: str, source_index: int | None = None) -> str:
    """Render one SQL statement in the official one-query-per-line format."""

    _ensure_official_import_paths()
    import sqlparse
    from sqlparse import tokens

    statements = tuple(
        statement for statement in sqlparse.parse(sql) if str(statement).strip()
    )
    label = "" if source_index is None else f" at source index {source_index}"
    if len(statements) != 1:
        raise EvaluationError(f"Spider prediction is not one statement{label}")
    parts: list[str] = []
    pending_space = False
    for token in statements[0].flatten():
        if token.is_whitespace or token.ttype in tokens.Comment:
            pending_space = True
            continue
        value = token.value
        if "\n" in value or "\r" in value:
            raise EvaluationError(
                f"Spider line format cannot preserve a token newline{label}"
            )
        if pending_space and parts:
            parts.append(" ")
        parts.append(value)
        pending_space = False
    rendered = "".join(parts).strip()
    if not rendered:
        raise EvaluationError(f"Spider prediction is empty after rendering{label}")
    return rendered


def _execute_spider_read_only(
    module: Any,
    database_path: str | Path,
    sql: str,
    timeout_seconds: float,
) -> _RawExecution:
    path = Path(database_path).resolve()
    if not path.is_file():
        return _RawExecution(None, f"database not found: {path}", False)
    query = module.replace_cur_year(str(sql))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=min(timeout_seconds, 5.0),
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_read_only_authorizer)
        connection.text_factory = lambda value: value.decode(errors="ignore")
        rows = [tuple(row) for row in connection.execute(query).fetchall()]
        return _RawExecution(rows, None, False)
    except sqlite3.Error as error:
        return _RawExecution(None, str(error), False)
    finally:
        if connection is not None:
            connection.close()


def _public_rows(rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[_public_cell(value) for value in row] for row in rows]


class Evaluator:
    """Evaluate attempts with the pinned official benchmark semantics."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        feedback_regime: str,
        *,
        evaluation_protocol: str | None = None,
        timeout_seconds: float = 15.0,
        reference_timeout_seconds: float | None = None,
    ):
        dataset_config.validate()
        if feedback_regime not in FEEDBACK_REGIMES:
            raise ValueError(f"unsupported feedback regime: {feedback_regime}")
        protocol = evaluation_protocol or EVALUATION_PROTOCOLS[dataset_config.name]
        if protocol != EVALUATION_PROTOCOLS[dataset_config.name]:
            raise ValueError(
                f"unsupported evaluation protocol for {dataset_config.name}: {protocol}"
            )
        protocol_timeout = (
            EVALUATION_TIMEOUT_SECONDS[dataset_config.name]
            if reference_timeout_seconds is None
            else reference_timeout_seconds
        )
        self.dataset_config = dataset_config
        self.feedback_regime = feedback_regime
        self.evaluation_protocol = protocol
        self.timeout_seconds = _validated_timeout_seconds(
            timeout_seconds,
            "timeout_seconds",
        )
        self.reference_timeout_seconds = _validated_timeout_seconds(
            protocol_timeout,
            "reference_timeout_seconds",
        )
        self.evaluation_protocol_hash = evaluation_protocol_hash(
            dataset_config.name,
            protocol,
            self.reference_timeout_seconds,
        )
        self._references: dict[str, str] | None = None
        identity = evaluation_protocol_identity(
            dataset_config.name,
            protocol,
            self.reference_timeout_seconds,
        )
        _verify_protocol_sources(dataset_config.name, identity)
        _ensure_official_import_paths()
        self._evaluate_confirmed_counted = {
            "spider": self._evaluate_spider_confirmed_counted,
            "bird": self._evaluate_bird_confirmed_counted,
        }[dataset_config.name]

    def _validate_example(self, example: QueryExample) -> None:
        if example.dataset_name != self.dataset_config.name:
            raise ValueError(
                f"example dataset {example.dataset_name!r} does not match "
                f"{self.dataset_config.name!r}"
            )

    def _load_references(self) -> dict[str, str]:
        if self._references is not None:
            return self._references
        with Path(self.dataset_config.examples_path).open(
            "r",
            encoding="utf-8",
        ) as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise EvaluationError("examples file must contain a JSON list")

        sql_field = "query" if self.dataset_config.name == "spider" else "SQL"
        references: dict[str, str] = {}
        for source_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise EvaluationError("examples file contains a non-object record")
            query_id = stable_query_id(
                self.dataset_config.name,
                record,
                source_index,
            )
            if query_id in references:
                raise EvaluationError(f"duplicate reference query_id: {query_id}")
            reference_sql = record.get(sql_field)
            if not isinstance(reference_sql, str) or not reference_sql.strip():
                raise EvaluationError(f"missing reference SQL for query_id={query_id}")
            references[query_id] = reference_sql.strip()
        self._references = references
        return references

    def _reference_sql(self, query_id: str) -> str:
        references = self._load_references()
        try:
            return references[str(query_id)]
        except KeyError as error:
            raise EvaluationError(f"no reference SQL for query_id={query_id}") from error

    @staticmethod
    def _execution_failure_outcome(
        result: ExecutionResult | _RawExecution,
        *,
        regime: str,
    ) -> Outcome:
        return Outcome(
            predicted_exec_ok=False,
            predicted_rows=None,
            db_error=result.error or "unknown execution error",
            oracle_correct=(
                False
                if regime == FeedbackRegime.DENOTATION_CONFIRMED.value
                else None
            ),
            status=(
                OutcomeStatus.TIMEOUT
                if result.timed_out
                else OutcomeStatus.EXECUTION_ERROR
            ),
            feedback_regime=regime,
        )

    def _evaluate_spider_confirmed_counted(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> CountedEvaluation:
        try:
            rendered = render_spider_prediction(predicted_sql, example.source_index)
        except EvaluationError as error:
            failed = ExecutionResult(None, str(error), False)
            return CountedEvaluation(
                self._execution_failure_outcome(
                    failed,
                    regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
                ),
                0,
            )
        module = _load_spider_exec_module()
        reference_sql = self._reference_sql(example.query_id)
        official_prediction = rendered.replace("value", "1")
        state: dict[str, Any] = {
            "call_index": 0,
            "db_executions": 0,
            "last_prediction": None,
        }

        async def execute_counted(
            sqlite_path: str,
            query: str,
            process_id: str = "",
            timeout: int = 60,
        ) -> tuple[str, Any]:
            del process_id, timeout
            is_gold = state["call_index"] % 2 == 0
            state["call_index"] += 1
            result = _execute_spider_read_only(
                module,
                sqlite_path,
                query,
                self.reference_timeout_seconds,
            )
            state["db_executions"] += 1
            if not is_gold:
                state["last_prediction"] = result
            if result.succeeded:
                return "result", result.rows or []
            error: Any = TimeoutError if result.timed_out else RuntimeError(result.error)
            return "exception", error

        with _SPIDER_MODULE_LOCK:
            original = module.exec_on_db
            module.exec_on_db = execute_counted
            try:
                score = module.eval_exec_match(
                    db=str(Path(example.database_path).resolve()),
                    p_str=official_prediction,
                    g_str=reference_sql,
                    plug_value=False,
                    keep_distinct=True,
                    progress_bar_for_each_datapoint=False,
                )
            except AssertionError as error:
                raise EvaluationError(
                    f"Spider reference execution failed for query_id={example.query_id}"
                ) from error
            finally:
                module.exec_on_db = original

        last_prediction = state["last_prediction"]
        if not isinstance(last_prediction, _RawExecution):
            raise EvaluationError("Spider evaluator did not execute the prediction")
        executions = int(state["db_executions"])
        if not last_prediction.succeeded:
            return CountedEvaluation(
                self._execution_failure_outcome(
                    last_prediction,
                    regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
                ),
                executions,
            )
        correct = score == 1
        outcome = Outcome(
            predicted_exec_ok=True,
            predicted_rows=_public_rows(last_prediction.rows or []),
            db_error=None,
            oracle_correct=correct,
            status=(
                OutcomeStatus.CORRECT
                if correct
                else OutcomeStatus.DENOTATION_MISMATCH
            ),
            feedback_regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
        )
        return CountedEvaluation(outcome, executions)

    def _evaluate_bird_confirmed_counted(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> CountedEvaluation:
        _ensure_official_import_paths()
        from func_timeout import FunctionTimedOut, func_timeout

        reference_sql = self._reference_sql(example.query_id)
        state: dict[str, Any] = {
            "db_executions": 0,
            "predicted_rows": None,
        }

        def execute_pair() -> bool:
            connection: sqlite3.Connection | None = None
            try:
                path = Path(example.database_path).resolve()
                connection = sqlite3.connect(
                    f"{path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=min(self.reference_timeout_seconds, 5.0),
                    check_same_thread=False,
                )
                connection.execute("PRAGMA query_only = ON")
                connection.set_authorizer(_read_only_authorizer)
                cursor = connection.cursor()
                state["db_executions"] = 1
                cursor.execute(str(predicted_sql).strip())
                predicted_rows = [tuple(row) for row in cursor.fetchall()]
                state["predicted_rows"] = predicted_rows
                state["db_executions"] = 2
                cursor.execute(reference_sql)
                reference_rows = [tuple(row) for row in cursor.fetchall()]
                return set(predicted_rows) == set(reference_rows)
            finally:
                if connection is not None:
                    connection.close()

        try:
            correct = bool(
                func_timeout(self.reference_timeout_seconds, execute_pair)
            )
        except FunctionTimedOut:
            predicted_rows = state["predicted_rows"]
            if predicted_rows is None:
                failed = ExecutionResult(None, "TIMEOUT", True)
                outcome = self._execution_failure_outcome(
                    failed,
                    regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
                )
            else:
                outcome = Outcome(
                    predicted_exec_ok=True,
                    predicted_rows=_public_rows(predicted_rows),
                    db_error=None,
                    oracle_correct=False,
                    status=OutcomeStatus.DENOTATION_MISMATCH,
                    feedback_regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
                )
            return CountedEvaluation(outcome, int(state["db_executions"]))
        except Exception as error:
            predicted_rows = state["predicted_rows"]
            if predicted_rows is None:
                failed = ExecutionResult(None, str(error), False)
                outcome = self._execution_failure_outcome(
                    failed,
                    regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
                )
            else:
                outcome = Outcome(
                    predicted_exec_ok=True,
                    predicted_rows=_public_rows(predicted_rows),
                    db_error=None,
                    oracle_correct=False,
                    status=OutcomeStatus.DENOTATION_MISMATCH,
                    feedback_regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
                )
            return CountedEvaluation(outcome, int(state["db_executions"]))

        predicted_rows = state["predicted_rows"]
        if predicted_rows is None:
            raise EvaluationError("BIRD evaluator completed without prediction rows")
        outcome = Outcome(
            predicted_exec_ok=True,
            predicted_rows=_public_rows(predicted_rows),
            db_error=None,
            oracle_correct=correct,
            status=(
                OutcomeStatus.CORRECT
                if correct
                else OutcomeStatus.DENOTATION_MISMATCH
            ),
            feedback_regime=FeedbackRegime.DENOTATION_CONFIRMED.value,
        )
        return CountedEvaluation(outcome, int(state["db_executions"]))

    def _evaluate_dbms_only_counted(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> CountedEvaluation:
        db_executions = int(Path(example.database_path).resolve().is_file())
        predicted = execute_read_only(
            example.database_path,
            predicted_sql,
            timeout_seconds=self.timeout_seconds,
        )
        if not predicted.succeeded:
            return CountedEvaluation(
                self._execution_failure_outcome(
                    predicted,
                    regime=FeedbackRegime.DBMS_ONLY.value,
                ),
                db_executions,
            )
        outcome = Outcome(
            predicted_exec_ok=True,
            predicted_rows=predicted.rows,
            db_error=None,
            oracle_correct=None,
            status=OutcomeStatus.DENOTATION_MISMATCH,
            feedback_regime=FeedbackRegime.DBMS_ONLY.value,
        )
        return CountedEvaluation(outcome, db_executions)

    def evaluate_counted(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> CountedEvaluation:
        """Return legal online feedback and exact physical SQL execution count."""

        self._validate_example(example)
        if self.feedback_regime == FeedbackRegime.DENOTATION_CONFIRMED.value:
            return self._evaluate_confirmed_counted(example, predicted_sql)
        return self._evaluate_dbms_only_counted(example, predicted_sql)

    def evaluate(self, example: QueryExample, predicted_sql: str) -> Outcome:
        """Compatibility API returning only the legal public outcome."""

        return self.evaluate_counted(example, predicted_sql).outcome

    def evaluate_offline_counted(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> CountedEvaluation:
        """Official oracle evaluation for post-episode metrics."""

        self._validate_example(example)
        return self._evaluate_confirmed_counted(example, predicted_sql)

    def evaluate_offline(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> Outcome:
        return self.evaluate_offline_counted(example, predicted_sql).outcome

    def offline_correctness(
        self,
        example: QueryExample,
        predicted_sql: str,
    ) -> bool:
        return self.evaluate_offline(example, predicted_sql).oracle_correct is True


def _parse_json_result(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise EvaluationError("official evaluator produced no JSON output")
    candidates = [stripped] + [
        line.strip()
        for line in reversed(stripped.splitlines())
        if line.strip()
    ]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise EvaluationError("official evaluator output did not contain a JSON object")


class OfficialEvaluatorAdapter:
    """Run an explicitly configured Spider/BIRD official evaluator command.

    Command tokens may contain ``{predictions}``, ``{gold}``, ``{tables}``,
    ``{database_root}``, and ``{output}`` placeholders. The command must emit a
    JSON object to stdout or write it to ``{output}``.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        command: Sequence[str],
        *,
        timeout_seconds: float = 3600.0,
    ):
        dataset_config.validate()
        if not command:
            raise ValueError("official evaluator command must be explicit")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not any("{predictions}" in token for token in command):
            raise ValueError("official evaluator command requires {predictions}")
        self.dataset_config = dataset_config
        self.command = tuple(str(token) for token in command)
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _substitute(token: str, replacements: Mapping[str, str]) -> str:
        rendered = token
        for placeholder, value in replacements.items():
            rendered = rendered.replace("{" + placeholder + "}", value)
        return rendered

    def evaluate(self, predictions_path: str | Path) -> dict[str, Any]:
        predictions = Path(predictions_path).resolve()
        if not predictions.is_file():
            raise FileNotFoundError(f"predictions file not found: {predictions}")

        with tempfile.TemporaryDirectory(prefix="merit-official-eval-") as temp_dir:
            output = Path(temp_dir) / "official_eval.json"
            replacements = {
                "predictions": str(predictions),
                "gold": str(Path(self.dataset_config.examples_path).resolve()),
                "tables": str(Path(self.dataset_config.tables_path).resolve()),
                "database_root": str(
                    Path(self.dataset_config.database_root).resolve()
                ),
                "output": str(output),
            }
            command = [
                self._substitute(token, replacements)
                for token in self.command
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise EvaluationError("official evaluator timed out") from exc
            if completed.returncode != 0:
                raise EvaluationError(
                    "official evaluator failed "
                    f"(exit={completed.returncode}): {completed.stderr.strip()}"
                )
            if output.is_file() and output.stat().st_size:
                result = _parse_json_result(
                    output.read_text(encoding="utf-8")
                )
            else:
                result = _parse_json_result(completed.stdout)
            result["official"] = True
            result.setdefault("dataset_name", self.dataset_config.name)
            return result


__all__ = [
    "CountedEvaluation",
    "EvaluationError",
    "Evaluator",
    "ExecutionResult",
    "FeedbackRegime",
    "OfficialEvaluatorAdapter",
    "Outcome",
    "OutcomeStatus",
    "render_spider_prediction",
    "execute_read_only",
]
