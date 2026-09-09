#!/usr/bin/env python3
"""Semantically audit one pinned BIRD official/internal execution disagreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_RELATIVE = Path(
    "official_evaluation/audit_bird_execution_reconciliation.py"
)
EXPECTED_RUN_RELATIVE = Path(
    "runs/bird/bird-iterative-official-v2-s1"
)
EXPECTED_CONFIG_EXAMPLES_RELATIVE = Path("data/BIRD/dev.json")
EXPECTED_CONFIG_DATABASE_ROOT_RELATIVE = Path("data/BIRD/dev_databases")
EXPECTED_EXAMPLES_RELATIVE = Path("data/dev_20240627/dev.json")
EXPECTED_DATABASE_ROOT_RELATIVE = Path("data/dev_20240627/dev_databases")
EXPECTED_DATABASE_RELATIVE = (
    EXPECTED_DATABASE_ROOT_RELATIVE / "card_games/card_games.sqlite"
)
EXPECTED_OFFICIAL_GOLD_RELATIVE = Path("data/dev_20240627/dev.sql")

FORMAT_VERSION = "merit-bird-execution-reconciliation-v1"
EXPECTED_SOURCE_HASH = (
    "21472d8034fb02780d4f67dd090d0df5a9f44fb8172088c781ab6818cfa0f6cc"
)
EXPECTED_PROTOCOL = "bird_damo_exec_483554e_30s_v1"
EXPECTED_PROTOCOL_HASH = (
    "36c49fe2eed680ad0dff8138fc5ae46a792fea392907e32ad145ff2bc1bd406c"
)
EXPECTED_DATABASE_SHA256 = (
    "c98bdb57fe7474da798b407785544b9af0daaad5d61fd21e2a73309493bc1227"
)
EXPECTED_TIMEOUT_SECONDS = 30.0
EXPECTED_TOTAL = 1534
EXPECTED_OFFICIAL_CORRECT = 726
EXPECTED_INTERNAL_CORRECT = 727
EXPECTED_IDENTITY = {
    "dataset": "bird",
    "method": "iterative",
    "run_id": "bird-iterative-official-v2-s1",
    "stream_seed": 1,
    "query_id": "518",
    "source_index": 518,
    "db_id": "card_games",
}
RUN_ARTIFACT_NAMES = (
    "config.json",
    "manifest.json",
    "metrics.json",
    "predictions.jsonl",
    "trajectories.jsonl",
    "official_eval.json",
)
OFFICIAL_IMMUTABLE_NAMES = RUN_ARTIFACT_NAMES[:5]
ARTIFACT_KEYS = (
    *RUN_ARTIFACT_NAMES,
    "database",
    "examples",
    "official_gold",
)
EXPECTED_EVALUATOR_OPTIONS = {
    "comparison": "set(predicted_rows) == set(gold_rows)",
    "data_mode": "dev",
    "meta_time_out_seconds": EXPECTED_TIMEOUT_SECONDS,
    "mode_gt": "gt",
    "mode_predict": "gpt",
    "num_cpus": 1,
}


class ReconciliationContractError(RuntimeError):
    """The target run or semantic audit differs from the pinned contract."""


@dataclass(frozen=True)
class PairExecution:
    predicted_rows: tuple[tuple[Any, ...], ...]
    gold_rows: tuple[tuple[Any, ...], ...]
    prediction_elapsed_seconds: float
    gold_elapsed_seconds: float
    pair_elapsed_seconds: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationContractError(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    _require(
        (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _resolved_repo_file(relative: Path, label: str) -> Path:
    _require(not relative.is_absolute(), f"{label} path must be repository-relative")
    _require(".." not in relative.parts, f"{label} path cannot traverse parents")
    root = REPOSITORY_ROOT.resolve()
    resolved = (root / relative).resolve(strict=True)
    _require(resolved.is_relative_to(root), f"{label} path leaves the repository")
    _require(resolved.is_file(), f"{label} file does not exist: {relative}")
    return resolved


def _expected_run_directory(run_directory: Path) -> Path:
    expected = (REPOSITORY_ROOT / EXPECTED_RUN_RELATIVE).resolve(strict=True)
    observed = run_directory.resolve(strict=True)
    _require(observed == expected, "run directory is not the pinned iterative seed-1 run")
    _require(observed.is_dir(), "pinned run path is not a directory")
    return observed


def _run_paths(run_directory: Path) -> dict[str, Path]:
    paths = {name: run_directory / name for name in RUN_ARTIFACT_NAMES}
    for name, path in paths.items():
        _require(path.is_file(), f"run artifact is missing: {name}")
    return paths


def _artifact_paths(run_directory: Path) -> dict[str, Path]:
    paths = _run_paths(run_directory)
    paths["database"] = _resolved_repo_file(
        EXPECTED_DATABASE_RELATIVE, "database"
    )
    paths["examples"] = _resolved_repo_file(EXPECTED_EXAMPLES_RELATIVE, "examples")
    paths["official_gold"] = _resolved_repo_file(
        EXPECTED_OFFICIAL_GOLD_RELATIVE, "official gold"
    )
    _require(tuple(paths) == ARTIFACT_KEYS, "audit artifact key order differs")
    return paths


def _artifact_relative_paths() -> dict[str, str]:
    relative = {
        name: (EXPECTED_RUN_RELATIVE / name).as_posix()
        for name in RUN_ARTIFACT_NAMES
    }
    relative["database"] = EXPECTED_DATABASE_RELATIVE.as_posix()
    relative["examples"] = EXPECTED_EXAMPLES_RELATIVE.as_posix()
    relative["official_gold"] = EXPECTED_OFFICIAL_GOLD_RELATIVE.as_posix()
    return relative


def _snapshot_artifacts(paths: Mapping[str, Path]) -> dict[str, str]:
    _require(tuple(paths) == ARTIFACT_KEYS, "input artifact set differs")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    _require(
        hashes["database"] == EXPECTED_DATABASE_SHA256,
        "card_games database differs from the pinned run input",
    )
    return hashes


def _assert_snapshot(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> None:
    _require(_snapshot_artifacts(paths) == expected, "audit input changed")


def _jsonl_target(path: Path, label: str) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            _require(bool(stripped), f"blank {label} row at line {line_number}")
            row = _require_mapping(json.loads(stripped), f"{label}[{line_number}]")
            count += 1
            if (
                str(row.get("query_id")) == EXPECTED_IDENTITY["query_id"]
                or row.get("source_index") == EXPECTED_IDENTITY["source_index"]
            ):
                matches.append(row)
    _require(count == EXPECTED_TOTAL, f"{label} count is not {EXPECTED_TOTAL}")
    _require(len(matches) == 1, f"{label} does not contain one exact target row")
    return matches[0]


def _validate_config_and_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    dataset = _require_mapping(config.get("dataset"), "config.dataset")
    expected_config = {
        "dataset": EXPECTED_IDENTITY["dataset"],
        "method_name": EXPECTED_IDENTITY["method"],
        "stream_order_seed": EXPECTED_IDENTITY["stream_seed"],
        "evaluation_protocol": EXPECTED_PROTOCOL,
        "reference_timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
    }
    observed_config = {
        "dataset": dataset.get("name"),
        "method_name": config.get("method_name"),
        "stream_order_seed": config.get("stream_order_seed"),
        "evaluation_protocol": config.get("evaluation_protocol"),
        "reference_timeout_seconds": config.get("reference_timeout_seconds"),
    }
    _require(observed_config == expected_config, "run config identity differs")
    _require(
        dataset.get("examples_path")
        == EXPECTED_CONFIG_EXAMPLES_RELATIVE.as_posix(),
        "configured examples path differs",
    )
    _require(
        dataset.get("database_root")
        == EXPECTED_CONFIG_DATABASE_ROOT_RELATIVE.as_posix(),
        "configured database root differs",
    )
    _require(
        _resolved_repo_file(
            Path(str(dataset["examples_path"])), "configured examples"
        )
        == _resolved_repo_file(EXPECTED_EXAMPLES_RELATIVE, "examples"),
        "configured examples symlink target differs",
    )
    _require(
        (
            REPOSITORY_ROOT / str(dataset["database_root"])
        ).resolve(strict=True)
        == (REPOSITORY_ROOT / EXPECTED_DATABASE_ROOT_RELATIVE).resolve(strict=True),
        "configured database-root symlink target differs",
    )
    expected_manifest = {
        "completed": True,
        "run_id": EXPECTED_IDENTITY["run_id"],
        "method_name": EXPECTED_IDENTITY["method"],
        "stream_order_seed": EXPECTED_IDENTITY["stream_seed"],
        "source_hash": EXPECTED_SOURCE_HASH,
        "evaluator_protocol": EXPECTED_PROTOCOL,
        "evaluator_protocol_hash": EXPECTED_PROTOCOL_HASH,
    }
    _require(
        {key: manifest.get(key) for key in expected_manifest} == expected_manifest,
        "run manifest identity differs",
    )
    _require(
        manifest.get("config_hash") == _canonical_hash(config),
        "config does not match manifest config_hash",
    )


def _validate_prediction_and_trajectory(
    prediction: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> str:
    expected = {
        "query_id": EXPECTED_IDENTITY["query_id"],
        "source_index": EXPECTED_IDENTITY["source_index"],
        "db_id": EXPECTED_IDENTITY["db_id"],
    }
    for label, row in (("prediction", prediction), ("trajectory", trajectory)):
        observed = {
            "query_id": str(row.get("query_id")),
            "source_index": row.get("source_index"),
            "db_id": row.get("db_id"),
        }
        _require(observed == expected, f"{label} target identity differs")
    predicted_sql = prediction.get("predicted_sql")
    _require(
        isinstance(predicted_sql, str) and bool(predicted_sql.strip()),
        "target prediction SQL is empty",
    )
    _require(trajectory.get("final_sql") == predicted_sql, "final SQL differs")
    _require(trajectory.get("final_correct") is True, "internal target is not correct")
    _require(
        trajectory.get("final_failure_type") is None,
        "internally correct target retains a failure type",
    )
    return predicted_sql


def _validate_examples(examples: Any) -> str:
    _require(
        isinstance(examples, list) and len(examples) == EXPECTED_TOTAL,
        "examples do not contain the complete BIRD development set",
    )
    target = _require_mapping(
        examples[EXPECTED_IDENTITY["source_index"]], "examples[518]"
    )
    observed = {
        "query_id": str(target.get("question_id")),
        "db_id": target.get("db_id"),
    }
    _require(
        observed
        == {
            "query_id": EXPECTED_IDENTITY["query_id"],
            "db_id": EXPECTED_IDENTITY["db_id"],
        },
        "gold target identity differs",
    )
    gold_sql = target.get("SQL")
    _require(
        isinstance(gold_sql, str) and bool(gold_sql.strip()),
        "target gold SQL is empty",
    )
    return gold_sql


def _validate_official_gold(path: Path, examples_gold_sql: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(
        len(lines) == EXPECTED_TOTAL,
        "official gold does not contain the complete BIRD development set",
    )
    fields = lines[EXPECTED_IDENTITY["source_index"]].rsplit("\t", 1)
    _require(len(fields) == 2, "official gold target is not SQL<TAB>db_id")
    gold_sql, db_id = fields
    _require(db_id == EXPECTED_IDENTITY["db_id"], "official gold db_id differs")
    _require(gold_sql == examples_gold_sql, "official and example gold SQL differ")
    return gold_sql


def _validate_official_result(
    official: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> None:
    expected_summary = {
        "official": True,
        "dataset_name": EXPECTED_IDENTITY["dataset"],
        "final_correct_count": EXPECTED_OFFICIAL_CORRECT,
        "internal_final_correct_count": EXPECTED_INTERNAL_CORRECT,
        "total_count": EXPECTED_TOTAL,
        "matches_internal": False,
    }
    _require(
        {key: official.get(key) for key in expected_summary} == expected_summary,
        "official evaluation summary differs",
    )
    _require(
        official.get("execution_accuracy")
        == EXPECTED_OFFICIAL_CORRECT / EXPECTED_TOTAL,
        "official execution accuracy differs",
    )
    evaluator = _require_mapping(official.get("evaluator"), "official.evaluator")
    _require(
        evaluator.get("protocol_hash") == EXPECTED_PROTOCOL_HASH,
        "official evaluator protocol hash differs",
    )
    _require(
        evaluator.get("commit") == "483554eae102996f5ec1f4feab4e78ef29c2a394",
        "official BIRD evaluator commit differs",
    )
    _require(
        evaluator.get("options") == EXPECTED_EVALUATOR_OPTIONS,
        "official evaluator options differ",
    )
    inputs = _require_mapping(official.get("inputs"), "official.inputs")
    database_input = _require_mapping(
        inputs.get("database"), "official.inputs.database"
    )
    configured_database = _require_mapping(
        inputs.get("configured_dataset_database"),
        "official.inputs.configured_dataset_database",
    )
    examples_input = _require_mapping(
        inputs.get("examples"), "official.inputs.examples"
    )
    gold_input = _require_mapping(inputs.get("gold"), "official.inputs.gold")
    expected_examples = _resolved_repo_file(EXPECTED_EXAMPLES_RELATIVE, "examples")
    expected_gold = _resolved_repo_file(
        EXPECTED_OFFICIAL_GOLD_RELATIVE, "official gold"
    )
    expected_database_root = (
        REPOSITORY_ROOT / EXPECTED_DATABASE_ROOT_RELATIVE
    ).resolve(strict=True)
    _require(
        Path(str(database_input.get("root"))).resolve()
        == expected_database_root,
        "official scorer database root differs",
    )
    _require(
        Path(str(configured_database.get("root"))).resolve()
        == expected_database_root,
        "configured database provenance root differs",
    )
    _require(
        Path(str(examples_input.get("path"))).resolve() == expected_examples
        and examples_input.get("sha256") == artifact_hashes["examples"],
        "official scorer examples provenance differs",
    )
    _require(
        Path(str(gold_input.get("path"))).resolve() == expected_gold
        and gold_input.get("sha256") == artifact_hashes["official_gold"],
        "official scorer gold provenance differs",
    )
    run = _require_mapping(official.get("run"), "official.run")
    for key in (
        "run_id",
        "source_hash",
        "config_hash",
        "dataset_checksum",
        "database_manifest_hash",
    ):
        _require(run.get(key) == manifest.get(key), f"official run {key} differs")
    _require(
        run.get("validated_live_source_hash") == EXPECTED_SOURCE_HASH,
        "official live source hash differs",
    )
    immutable = _require_mapping(
        run.get("immutable_artifact_sha256"),
        "official.run.immutable_artifact_sha256",
    )
    _require(
        set(immutable) == set(OFFICIAL_IMMUTABLE_NAMES),
        "official immutable artifact set differs",
    )
    _require(
        all(immutable[name] == artifact_hashes[name] for name in immutable),
        "official immutable artifact hash differs",
    )
    per_query = official.get("per_query")
    _require(
        isinstance(per_query, list) and len(per_query) == EXPECTED_TOTAL,
        "official per-query vector is incomplete",
    )
    comparison_labels = {
        (True, True): "both_correct",
        (True, False): "internal_only",
        (False, True): "official_only",
        (False, False): "both_incorrect",
    }
    observed_counts: Counter[str] = Counter()
    mismatches: list[Mapping[str, Any]] = []
    for index, value in enumerate(per_query):
        row = _require_mapping(value, f"official.per_query[{index}]")
        internal_correct = row.get("internal_correct")
        official_correct = row.get("official_correct")
        correct = row.get("correct")
        _require(
            type(internal_correct) is bool
            and type(official_correct) is bool
            and type(correct) is bool,
            f"official per-query correctness is not boolean at {index}",
        )
        _require(correct is official_correct, f"official correct alias differs at {index}")
        label = comparison_labels[(internal_correct, official_correct)]
        _require(row.get("mismatch_class") == label, f"mismatch class differs at {index}")
        observed_counts[label] += 1
        if internal_correct is not official_correct:
            mismatches.append(row)
    _require(len(mismatches) == 1, "official mismatch vector is not a singleton")
    mismatch = mismatches[0]
    expected_mismatch = {
        "query_id": EXPECTED_IDENTITY["query_id"],
        "source_index": EXPECTED_IDENTITY["source_index"],
        "db_id": EXPECTED_IDENTITY["db_id"],
        "internal_correct": True,
        "official_correct": False,
        "mismatch_class": "internal_only",
    }
    _require(
        {key: mismatch.get(key) for key in expected_mismatch} == expected_mismatch,
        "official mismatch identity differs",
    )
    counts = official.get("correctness_comparison_counts")
    expected_counts = {
        "both_correct": EXPECTED_OFFICIAL_CORRECT,
        "both_incorrect": EXPECTED_TOTAL - EXPECTED_INTERNAL_CORRECT,
        "internal_only": 1,
        "official_only": 0,
    }
    _require(counts == expected_counts, "official correctness comparison counts differ")
    _require(
        {key: observed_counts[key] for key in expected_counts} == expected_counts,
        "official correctness comparison counts differ",
    )


def _validate_run(
    paths: Mapping[str, Path], artifact_hashes: Mapping[str, str]
) -> tuple[str, str]:
    config = _require_mapping(_read_json(paths["config.json"]), "config.json")
    manifest = _require_mapping(_read_json(paths["manifest.json"]), "manifest.json")
    metrics = _require_mapping(_read_json(paths["metrics.json"]), "metrics.json")
    official = _require_mapping(
        _read_json(paths["official_eval.json"]), "official_eval.json"
    )
    _validate_config_and_manifest(config, manifest)
    _require(
        metrics.get("total_examples") == EXPECTED_TOTAL
        and metrics.get("final_correct_count") == EXPECTED_INTERNAL_CORRECT,
        "internal metrics summary differs",
    )
    prediction = _jsonl_target(paths["predictions.jsonl"], "prediction")
    trajectory = _jsonl_target(paths["trajectories.jsonl"], "trajectory")
    predicted_sql = _validate_prediction_and_trajectory(prediction, trajectory)
    examples_gold_sql = _validate_examples(_read_json(paths["examples"]))
    gold_sql = _validate_official_gold(
        paths["official_gold"], examples_gold_sql
    )
    _validate_official_result(official, manifest, artifact_hashes)
    return predicted_sql, gold_sql


def _execute_pair(
    database: Path,
    predicted_sql: str,
    gold_sql: str,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> PairExecution:
    database_uri = f"{database.resolve().as_uri()}?mode=ro"
    pair_started = clock()
    connection = sqlite3.connect(
        database_uri,
        uri=True,
        timeout=5.0,
    )
    try:
        cursor = connection.cursor()
        prediction_started = clock()
        cursor.execute(predicted_sql)
        predicted_rows_native = cursor.fetchall()
        prediction_finished = clock()
        gold_started = clock()
        cursor.execute(gold_sql)
        gold_rows_native = cursor.fetchall()
        gold_finished = clock()
        set(predicted_rows_native) == set(gold_rows_native)
        pair_finished = clock()
    finally:
        connection.close()
    predicted_rows = tuple(tuple(row) for row in predicted_rows_native)
    gold_rows = tuple(tuple(row) for row in gold_rows_native)
    return PairExecution(
        predicted_rows=predicted_rows,
        gold_rows=gold_rows,
        prediction_elapsed_seconds=prediction_finished - prediction_started,
        gold_elapsed_seconds=gold_finished - gold_started,
        pair_elapsed_seconds=pair_finished - pair_started,
    )


def _canonical_cell(value: Any) -> list[Any]:
    if value is None:
        return ["null", None]
    if isinstance(value, (bool, int)):
        return ["number", int(value), 1]
    if isinstance(value, float):
        _require(math.isfinite(value), "SQLite result contains a non-finite float")
        numerator, denominator = value.as_integer_ratio()
        return ["number", numerator, denominator]
    if isinstance(value, str):
        return ["text", value]
    if isinstance(value, bytes):
        return ["blob", value.hex()]
    raise ReconciliationContractError(
        f"unsupported SQLite result type: {type(value).__name__}"
    )


def _canonical_row(value: tuple[Any, ...]) -> str:
    return json.dumps(
        [_canonical_cell(cell) for cell in value],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _row_set_summary(rows: Sequence[tuple[Any, ...]]) -> tuple[int, str]:
    canonical_rows = sorted({_canonical_row(tuple(row)) for row in rows})
    return len(canonical_rows), _canonical_hash(canonical_rows)


def _validate_timing_observation(evidence: Mapping[str, Any]) -> None:
    timing_names = (
        "prediction_elapsed_seconds",
        "gold_elapsed_seconds",
        "pair_elapsed_seconds",
    )
    timings = tuple(evidence.get(name) for name in timing_names)
    _require(
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0.0
            for value in timings
        ),
        "query timing is not positive and finite",
    )
    prediction_elapsed, gold_elapsed, pair_elapsed = timings
    component_elapsed = prediction_elapsed + gold_elapsed
    unclassified_elapsed = pair_elapsed - component_elapsed
    tolerance = max(0.25, pair_elapsed * 0.02)
    _require(
        -1e-9 <= unclassified_elapsed <= tolerance,
        "pair timing is inconsistent with component timings",
    )
    _require(
        type(evidence.get("timing_observation_count")) is int
        and evidence["timing_observation_count"] == 1,
        "timing observation count differs",
    )
    _require(
        evidence.get("timing_used_as_acceptance_gate") is False,
        "timing was incorrectly marked as an acceptance gate",
    )
    expected_exceeded = pair_elapsed > EXPECTED_TIMEOUT_SECONDS
    _require(
        evidence.get("pair_exceeded_protocol_timeout") is expected_exceeded,
        "protocol-threshold flag is inconsistent with observed pair timing",
    )


def _evidence(execution: PairExecution) -> dict[str, Any]:
    predicted_set = set(execution.predicted_rows)
    gold_set = set(execution.gold_rows)
    _require(predicted_set == gold_set, "prediction and gold row sets differ")
    predicted_distinct, predicted_hash = _row_set_summary(
        execution.predicted_rows
    )
    gold_distinct, gold_hash = _row_set_summary(execution.gold_rows)
    _require(
        predicted_distinct == gold_distinct and predicted_hash == gold_hash,
        "canonical row-set evidence differs",
    )
    evidence = {
        "prediction_row_count": len(execution.predicted_rows),
        "gold_row_count": len(execution.gold_rows),
        "prediction_distinct_row_count": predicted_distinct,
        "gold_distinct_row_count": gold_distinct,
        "prediction_row_set_sha256": predicted_hash,
        "gold_row_set_sha256": gold_hash,
        "row_sets_equal": True,
        "prediction_elapsed_seconds": execution.prediction_elapsed_seconds,
        "gold_elapsed_seconds": execution.gold_elapsed_seconds,
        "pair_elapsed_seconds": execution.pair_elapsed_seconds,
        "pair_exceeded_protocol_timeout": (
            execution.pair_elapsed_seconds > EXPECTED_TIMEOUT_SECONDS
        ),
        "timing_observation_count": 1,
        "timing_used_as_acceptance_gate": False,
    }
    _validate_timing_observation(evidence)
    return evidence


def _explicit_output(path: Path, input_paths: Mapping[str, Path]) -> Path:
    _require(not os.path.lexists(path), f"output already exists: {path}")
    _require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    root = REPOSITORY_ROOT.resolve()
    output = (path.parent.resolve() / path.name).resolve(strict=False)
    _require(output.is_relative_to(root), "output path leaves the repository")
    _require(
        output not in {value.resolve() for value in input_paths.values()},
        "output path aliases an audit input",
    )
    return output


def _atomic_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _require(not os.path.lexists(path), f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReconciliationContractError(
                f"output already exists: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def run_audit(run_directory: Path, output_path: Path) -> Mapping[str, Any]:
    run = _expected_run_directory(run_directory)
    paths = _artifact_paths(run)
    output = _explicit_output(output_path, paths)
    generator = _resolved_repo_file(GENERATOR_RELATIVE, "generator")
    generator_sha256 = _sha256_file(generator)
    artifact_hashes = _snapshot_artifacts(paths)
    predicted_sql, gold_sql = _validate_run(paths, artifact_hashes)
    execution = _execute_pair(paths["database"], predicted_sql, gold_sql)
    evidence = _evidence(execution)
    _assert_snapshot(paths, artifact_hashes)
    _require(_sha256_file(generator) == generator_sha256, "generator changed")
    relative_paths = _artifact_relative_paths()
    result = {
        "format_version": FORMAT_VERSION,
        "canonical_reporting_source": "official",
        "identity": dict(EXPECTED_IDENTITY),
        "protocol": {
            "name": EXPECTED_PROTOCOL,
            "hash": EXPECTED_PROTOCOL_HASH,
            "timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
        },
        "mismatch": {
            "internal_correct": True,
            "official_correct": False,
            "class": "internal_only",
        },
        "artifacts": {
            name: {
                "path": relative_paths[name],
                "sha256": artifact_hashes[name],
            }
            for name in ARTIFACT_KEYS
        },
        "sql": {
            "prediction_sha256": _sha256_text(predicted_sql),
            "gold_sha256": _sha256_text(gold_sql),
        },
        "evidence": evidence,
        "generator": {
            "path": GENERATOR_RELATIVE.as_posix(),
            "sha256": generator_sha256,
        },
    }
    _atomic_json_exclusive(output, result)
    try:
        _assert_snapshot(paths, artifact_hashes)
        _require(_sha256_file(generator) == generator_sha256, "generator changed")
    except ReconciliationContractError:
        output.unlink(missing_ok=True)
        raise
    print(f"Wrote {output}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", help="exact completed BIRD run directory")
    parser.add_argument(
        "--output",
        required=True,
        help="explicit new reconciliation JSON path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_audit(Path(args.run_directory), Path(args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ReconciliationContractError,
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ) as error:
        print(f"BIRD execution reconciliation error: {error}", file=os.sys.stderr)
        raise SystemExit(1) from error
