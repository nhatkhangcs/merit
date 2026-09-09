#!/usr/bin/env python3
"""Evaluate a completed MERIT run with pinned official Spider/BIRD code."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
VENDOR_PYTHON = VENDOR_ROOT / "python"
VENDOR_NLTK_DATA = VENDOR_ROOT / "nltk_data"

SPIDER_REPOSITORY = "https://github.com/taoyds/test-suite-sql-eval"
SPIDER_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
SPIDER_ARCHIVE_SHA256 = (
    "9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96"
)
SPIDER_SOURCE_SHA256 = {
    "evaluation.py": "7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779",
    "exec_eval.py": "29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7",
    "exec_subprocess.py": "1366694f8ad4d80cdd8fb45eb8c34f48f101fa88c3e0792bd3519f6bbe8530d9",
    "parse.py": "ef04211a6e1c1e142571157f5c1999613e3451084c044083b2de1977f1f622c5",
    "process_sql.py": "927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1",
}

BIRD_REPOSITORY = "https://github.com/AlibabaResearch/DAMO-ConvAI"
BIRD_COMMIT = "483554eae102996f5ec1f4feab4e78ef29c2a394"
BIRD_SOURCE_SHA256 = {
    "evaluation.py": "2f591e559dc2d97e5b35d5b656e80b0c2edf968f0bb5a78ddfd1d88b4bbbc472",
    "run_evaluation.sh": "b3b3dc9daa06549b4818c38a320fde32d08fe8fc0fbac46d244267322a6cfbc5",
}

IMMUTABLE_RUN_FILES = (
    "config.json",
    "manifest.json",
    "metrics.json",
    "predictions.jsonl",
    "trajectories.jsonl",
)
EXPECTED_VENDOR_PACKAGES = {
    "click": "8.4.2",
    "func-timeout": "4.3.5",
    "joblib": "1.5.3",
    "nltk": "3.9.1",
    "regex": "2026.7.19",
    "sqlparse": "0.5.3",
    "tqdm": "4.67.0",
}
BIRD_DELIMITER = "\t----- bird -----\t"
CORRECTNESS_LABELS = {
    (True, True): "both_correct",
    (True, False): "internal_only",
    (False, True): "official_only",
    (False, False): "both_incorrect",
}


class EvaluationContractError(RuntimeError):
    """The completed run or official evaluator violates the evaluation contract."""


@dataclass(frozen=True)
class ExpectedExample:
    source_index: int
    query_id: str
    db_id: str
    difficulty: str
    reference_sql: str


@dataclass(frozen=True)
class RunContext:
    run_directory: Path
    dataset_name: str
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    examples_path: Path
    tables_path: Path
    database_root: Path
    expected_examples: tuple[ExpectedExample, ...]
    predictions_source_order: tuple[Mapping[str, Any], ...]
    trajectories_source_order: tuple[Mapping[str, Any], ...]
    immutable_hashes: Mapping[str, str]
    live_source_hash: str
    configured_database_manifest: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise EvaluationContractError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _live_source_hash() -> str:
    paths = sorted((REPOSITORY_ROOT / "merit").glob("*.py")) + sorted(
        (REPOSITORY_ROOT / "scripts").glob("*.py")
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
        if (path := REPOSITORY_ROOT / relative).is_file()
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _configured_evaluation_protocol(
    config: Mapping[str, Any],
    dataset_name: str,
) -> tuple[dict[str, Any], str]:
    repository_path = str(REPOSITORY_ROOT)
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    from merit.config import evaluation_protocol_hash, evaluation_protocol_identity

    try:
        protocol = config["evaluation_protocol"]
        timeout_seconds = config["reference_timeout_seconds"]
        identity = evaluation_protocol_identity(
            dataset_name, protocol, timeout_seconds
        )
        protocol_hash = evaluation_protocol_hash(
            dataset_name, protocol, timeout_seconds
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationContractError(
            "run config has an invalid evaluation protocol identity"
        ) from error
    return identity, protocol_hash


def _configured_database_manifest(root: Path) -> dict[str, Any]:
    paths = sorted(
        root.rglob("*.sqlite"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not paths:
        raise EvaluationContractError(f"no SQLite files found under {root}")
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root)
        size = path.stat().st_size
        entries.append(
            {
                "db_id": relative.parts[0] if len(relative.parts) > 1 else path.stem,
                "relative_path": relative.as_posix(),
                "size_bytes": size,
                "sha256": _sha256_file(path),
            }
        )
        total_bytes += size
    return {
        "root": str(root),
        "sqlite_file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256": _canonical_hash(entries),
        "algorithm": "canonical SHA256 of MERIT DatabaseManifestEntry objects",
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise EvaluationContractError(
                    f"blank JSONL record at {path}:{line_number}"
                )
            value = json.loads(stripped)
            if not isinstance(value, Mapping):
                raise EvaluationContractError(
                    f"JSONL record is not an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{label} must be a JSON object")
    return value


def _require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationContractError(f"{label} must be an integer")
    return value


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _artifact_hashes(run_directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in IMMUTABLE_RUN_FILES:
        path = run_directory / name
        if not path.is_file():
            raise EvaluationContractError(f"completed run artifact is missing: {path}")
        hashes[name] = _sha256_file(path)
    return hashes


def _assert_artifacts_unchanged(
    run_directory: Path, expected_hashes: Mapping[str, str]
) -> None:
    observed = _artifact_hashes(run_directory)
    if observed != expected_hashes:
        changed = sorted(
            name
            for name in set(expected_hashes) | set(observed)
            if expected_hashes.get(name) != observed.get(name)
        )
        raise EvaluationContractError(
            "immutable run artifacts changed during official evaluation: "
            + ", ".join(changed)
        )


def _stable_query_id(
    dataset_name: str, record: Mapping[str, Any], source_index: int
) -> str:
    if dataset_name not in {"spider", "bird"}:
        raise EvaluationContractError(f"unsupported dataset: {dataset_name}")
    question_id = record.get("question_id")
    return str(question_id) if question_id is not None else str(source_index)


def _load_completed_run(run_directory: Path) -> RunContext:
    directory = run_directory.resolve()
    if not directory.is_dir():
        raise EvaluationContractError(f"run directory does not exist: {directory}")
    immutable_hashes = _artifact_hashes(directory)
    config = _require_mapping(_read_json(directory / "config.json"), "config.json")
    manifest = _require_mapping(
        _read_json(directory / "manifest.json"), "manifest.json"
    )
    metrics = _require_mapping(_read_json(directory / "metrics.json"), "metrics.json")
    if manifest.get("completed") is not True:
        raise EvaluationContractError("manifest.json does not mark the run completed")
    if manifest.get("config_hash") != _canonical_hash(config):
        raise EvaluationContractError("config.json does not match manifest config_hash")

    dataset = _require_mapping(config.get("dataset"), "config.dataset")
    dataset_name = str(dataset.get("name", "")).strip().lower()
    if dataset_name not in {"spider", "bird"}:
        raise EvaluationContractError(f"unsupported dataset: {dataset_name}")
    _, protocol_hash = _configured_evaluation_protocol(config, dataset_name)
    if manifest.get("evaluator_protocol") != config.get("evaluation_protocol"):
        raise EvaluationContractError(
            "manifest evaluator protocol differs from config"
        )
    if manifest.get("evaluator_protocol_hash") != protocol_hash:
        raise EvaluationContractError(
            "manifest evaluator protocol hash differs from config"
        )
    examples_path = _resolve_repository_path(str(dataset.get("examples_path", "")))
    tables_path = _resolve_repository_path(str(dataset.get("tables_path", "")))
    database_root = _resolve_repository_path(str(dataset.get("database_root", "")))
    for path, label in (
        (examples_path, "examples"),
        (tables_path, "tables"),
        (database_root, "database root"),
    ):
        if not path.exists():
            raise EvaluationContractError(f"{label} path does not exist: {path}")

    raw_examples = _read_json(examples_path)
    if not isinstance(raw_examples, list) or not all(
        isinstance(record, Mapping) for record in raw_examples
    ):
        raise EvaluationContractError("dataset examples must be a JSON list of objects")
    expected_examples = tuple(
        ExpectedExample(
            source_index=source_index,
            query_id=_stable_query_id(dataset_name, record, source_index),
            db_id=str(record.get("db_id", "")).strip(),
            difficulty=str(
                record.get("difficulty")
                if dataset_name == "bird"
                else record.get("hardness", "unknown")
            ).strip()
            or "unknown",
            reference_sql=str(
                record.get("query" if dataset_name == "spider" else "SQL", "")
            ).strip(),
        )
        for source_index, record in enumerate(raw_examples)
    )
    if any(not example.db_id for example in expected_examples):
        raise EvaluationContractError("dataset contains an empty db_id")
    if any(not example.reference_sql for example in expected_examples):
        raise EvaluationContractError("dataset contains empty reference SQL")

    examples_sha256 = _sha256_file(examples_path)
    tables_sha256 = _sha256_file(tables_path)
    dataset_checksum = _canonical_hash(
        {
            "dataset_name": dataset_name,
            "examples_sha256": examples_sha256,
            "tables_sha256": tables_sha256,
        }
    )
    if manifest.get("dataset_checksum") != dataset_checksum:
        raise EvaluationContractError(
            "configured dataset files do not match manifest dataset_checksum"
        )
    live_source_hash = _live_source_hash()
    if manifest.get("source_hash") != live_source_hash:
        raise EvaluationContractError(
            "live MERIT source does not match manifest source_hash"
        )
    configured_database_manifest = _configured_database_manifest(database_root)
    if (
        manifest.get("database_manifest_hash")
        != configured_database_manifest["sha256"]
    ):
        raise EvaluationContractError(
            "configured databases do not match manifest database_manifest_hash"
        )

    predictions = _read_jsonl(directory / "predictions.jsonl")
    trajectories = _read_jsonl(directory / "trajectories.jsonl")
    total = len(expected_examples)
    if len(predictions) != total or len(trajectories) != total:
        raise EvaluationContractError(
            "completed prediction/trajectory counts do not cover the dataset"
        )
    if _require_integer(metrics.get("total_examples"), "metrics.total_examples") != total:
        raise EvaluationContractError("metrics total_examples does not match the dataset")

    stream = list(expected_examples)
    stream_seed = _require_integer(
        config.get("stream_order_seed"), "config.stream_order_seed"
    )
    random.Random(stream_seed).shuffle(stream)
    identity_fields = ("query_id", "db_id", "source_index", "stream_position")
    for position, (prediction, trajectory, expected) in enumerate(
        zip(predictions, trajectories, stream)
    ):
        observed_identity = (
            str(prediction.get("query_id")),
            str(prediction.get("db_id")),
            _require_integer(
                prediction.get("source_index"),
                f"prediction[{position}].source_index",
            ),
            _require_integer(
                prediction.get("stream_position"),
                f"prediction[{position}].stream_position",
            ),
        )
        expected_identity = (
            expected.query_id,
            expected.db_id,
            expected.source_index,
            position,
        )
        if observed_identity != expected_identity:
            raise EvaluationContractError(
                f"prediction stream identity mismatch at position {position}"
            )
        if any(
            prediction.get(field) != trajectory.get(field)
            for field in identity_fields
        ):
            raise EvaluationContractError(
                f"prediction/trajectory identity mismatch at position {position}"
            )
        sql = prediction.get("predicted_sql")
        if not isinstance(sql, str) or not sql.strip():
            raise EvaluationContractError(
                f"prediction SQL is empty at stream position {position}"
            )
        if trajectory.get("final_sql") != sql:
            raise EvaluationContractError(
                f"prediction SQL differs from trajectory at position {position}"
            )

    derived_correct = sum(bool(row.get("final_correct")) for row in trajectories)
    internal_correct = _require_integer(
        metrics.get("final_correct_count"), "metrics.final_correct_count"
    )
    if derived_correct != internal_correct:
        raise EvaluationContractError(
            "metrics final_correct_count does not match trajectories"
        )
    predictions_source_order = tuple(
        sorted(predictions, key=lambda row: int(row["source_index"]))
    )
    trajectories_source_order = tuple(
        sorted(trajectories, key=lambda row: int(row["source_index"]))
    )
    if tuple(int(row["source_index"]) for row in predictions_source_order) != tuple(
        range(total)
    ):
        raise EvaluationContractError("prediction source_index coverage is not exact")
    for prediction, expected in zip(predictions_source_order, expected_examples):
        if (
            str(prediction["query_id"]),
            str(prediction["db_id"]),
        ) != (expected.query_id, expected.db_id):
            raise EvaluationContractError(
                f"source-order prediction mismatch at index {expected.source_index}"
            )

    _assert_artifacts_unchanged(directory, immutable_hashes)
    return RunContext(
        run_directory=directory,
        dataset_name=dataset_name,
        config=config,
        manifest=manifest,
        metrics=metrics,
        examples_path=examples_path,
        tables_path=tables_path,
        database_root=database_root,
        expected_examples=expected_examples,
        predictions_source_order=predictions_source_order,
        trajectories_source_order=trajectories_source_order,
        immutable_hashes=immutable_hashes,
        live_source_hash=live_source_hash,
        configured_database_manifest=configured_database_manifest,
    )


def _validate_gold(
    path: Path, expected_examples: Sequence[ExpectedExample]
) -> tuple[str, ...]:
    if not path.is_file():
        raise EvaluationContractError(f"gold file does not exist: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected_examples):
        raise EvaluationContractError(
            f"gold count {len(lines)} does not match dataset count "
            f"{len(expected_examples)}"
        )
    sqls: list[str] = []
    for expected, line in zip(expected_examples, lines):
        parts = line.strip().split("\t")
        if len(parts) != 2:
            raise EvaluationContractError(
                f"gold line {expected.source_index + 1} is not SQL<TAB>db_id"
            )
        sql, db_id = parts
        if not sql or db_id != expected.db_id:
            raise EvaluationContractError(
                f"gold identity mismatch at source index {expected.source_index}"
            )
        if sql.strip() != expected.reference_sql.strip():
            raise EvaluationContractError(
                f"gold SQL differs from configured reference at source index "
                f"{expected.source_index}"
            )
        sqls.append(sql)
    return tuple(sqls)


def _vendor_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(VENDOR_PYTHON)]):
        name = str(distribution.metadata["Name"]).lower().replace("_", "-")
        versions[name] = distribution.version
    observed = {name: versions.get(name, "") for name in EXPECTED_VENDOR_PACKAGES}
    if observed != EXPECTED_VENDOR_PACKAGES:
        raise EvaluationContractError(
            f"vendored dependency mismatch: expected={EXPECTED_VENDOR_PACKAGES}, "
            f"observed={observed}"
        )
    return observed


def _verify_sources(
    directory: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    observed = {
        relative: _sha256_file(directory / relative)
        for relative in sorted(expected_hashes)
    }
    if observed != dict(sorted(expected_hashes.items())):
        raise EvaluationContractError(
            f"vendored official evaluator source mismatch under {directory}"
        )
    return observed


def _tree_layout_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise EvaluationContractError(f"database root does not exist: {root}")
    paths = sorted(
        root.rglob("*.sqlite"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not paths:
        raise EvaluationContractError(f"no SQLite files found under {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        total_bytes += size
    return {
        "root": str(root),
        "sqlite_file_count": len(paths),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "algorithm": "sha256(relative_path NUL size NUL)",
    }


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluationContractError(f"cannot import official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_vendor_python() -> None:
    vendor_python = str(VENDOR_PYTHON)
    if vendor_python not in sys.path:
        sys.path.insert(0, vendor_python)
    os.environ["NLTK_DATA"] = str(VENDOR_NLTK_DATA)


def _official_provenance(
    context: RunContext,
    *,
    repository: str,
    commit: str,
    source_hashes: Mapping[str, str],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    identity, protocol_hash = _configured_evaluation_protocol(
        context.config, context.dataset_name
    )
    observed = {
        "repository": repository,
        "commit": commit,
        "source_sha256": dict(source_hashes),
        "options": dict(options),
        "dependencies": _vendor_package_versions(),
    }
    expected = {field: identity[field] for field in observed}
    if observed != expected:
        raise EvaluationContractError(
            "official evaluator provenance differs from configured protocol"
        )
    return {
        **observed,
        "protocol_hash": protocol_hash,
        "python_version": sys.version.split()[0],
    }


def _base_result(
    context: RunContext,
    *,
    final_correct_count: int,
    evaluator: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(context.expected_examples)
    internal = int(context.metrics["final_correct_count"])
    return {
        "official": True,
        "dataset_name": context.dataset_name,
        "score_name": (
            "test_suite_execution_accuracy"
            if context.dataset_name == "spider"
            else "execution_accuracy"
        ),
        "final_correct_count": final_correct_count,
        "total_count": total,
        "execution_accuracy": final_correct_count / total,
        "internal_final_correct_count": internal,
        "matches_internal": final_correct_count == internal,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "run_id": context.manifest.get("run_id"),
            "source_hash": context.manifest.get("source_hash"),
            "validated_live_source_hash": context.live_source_hash,
            "config_hash": context.manifest.get("config_hash"),
            "dataset_checksum": context.manifest.get("dataset_checksum"),
            "database_manifest_hash": context.manifest.get(
                "database_manifest_hash"
            ),
            "immutable_artifact_sha256": dict(context.immutable_hashes),
        },
        "inputs": {
            "configured_dataset_database": dict(
                context.configured_database_manifest
            ),
            **dict(inputs),
        },
        "evaluator": dict(evaluator),
    }


def _bird_default_gold(context: RunContext) -> Path:
    return context.examples_path.parent / "dev.sql"


def _evaluate_bird(
    context: RunContext,
    *,
    gold_path: Path,
    num_cpus: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if num_cpus < 1:
        raise EvaluationContractError("BIRD num_cpus must be positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise EvaluationContractError("BIRD timeout must be positive and finite")
    gold_sqls = _validate_gold(gold_path, context.expected_examples)
    source_hashes = _verify_sources(VENDOR_ROOT / "bird", BIRD_SOURCE_SHA256)
    _prepare_vendor_python()
    module = _load_module(
        "merit_pinned_bird_evaluation", VENDOR_ROOT / "bird" / "evaluation.py"
    )

    with tempfile.TemporaryDirectory(prefix="merit-bird-official-") as temp_name:
        temporary = Path(temp_name)
        prediction_directory = temporary / "predicted"
        gold_directory = temporary / "gold"
        prediction_directory.mkdir()
        gold_directory.mkdir()
        native_prediction = prediction_directory / "predict_dev.json"
        native_gold = gold_directory / "dev_gold.sql"
        native_values = {
            str(index): (
                str(prediction["predicted_sql"])
                + BIRD_DELIMITER
                + expected.db_id
            )
            for index, (prediction, expected) in enumerate(
                zip(
                    context.predictions_source_order,
                    context.expected_examples,
                )
            )
        }
        if any(
            BIRD_DELIMITER in str(prediction["predicted_sql"])
            for prediction in context.predictions_source_order
        ):
            raise EvaluationContractError(
                "BIRD prediction contains the official native-format delimiter"
            )
        native_prediction.write_text(
            json.dumps(native_values, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        native_gold.write_bytes(gold_path.read_bytes())
        native_hashes = {
            "predictions": _sha256_file(native_prediction),
            "gold": _sha256_file(native_gold),
        }

        root_argument = str(context.database_root) + os.sep
        prediction_prefix = str(prediction_directory) + os.sep
        gold_prefix = str(gold_directory) + os.sep
        predicted_queries, database_paths = module.package_sqls(
            prediction_prefix, root_argument, mode="gpt", data_mode="dev"
        )
        official_gold_queries, gold_database_paths = module.package_sqls(
            gold_prefix, root_argument, mode="gt", data_mode="dev"
        )
        expected_predictions = [
            str(row["predicted_sql"]) for row in context.predictions_source_order
        ]
        if predicted_queries != expected_predictions:
            raise EvaluationContractError(
                "BIRD native prediction packaging changed SQL or source order"
            )
        if official_gold_queries != list(gold_sqls):
            raise EvaluationContractError(
                "BIRD native gold packaging changed SQL or source order"
            )
        if database_paths != gold_database_paths:
            raise EvaluationContractError(
                "BIRD prediction and gold database paths differ"
            )

        module.exec_result = []
        module.run_sqls_parallel(
            list(zip(predicted_queries, official_gold_queries)),
            db_places=database_paths,
            num_cpus=num_cpus,
            meta_time_out=timeout_seconds,
        )
        official_results = module.sort_results(module.exec_result)
        if [row.get("sql_idx") for row in official_results] != list(
            range(len(context.expected_examples))
        ):
            raise EvaluationContractError(
                "BIRD official evaluator returned incomplete or duplicate results"
            )
        result_values = [
            _require_integer(row.get("res"), f"BIRD result[{index}].res")
            for index, row in enumerate(official_results)
        ]
        if any(value not in {0, 1} for value in result_values):
            raise EvaluationContractError("BIRD official result is not binary")
        final_correct_count = sum(result_values)
        per_query = []
        for expected, trajectory, official_result in zip(
            context.expected_examples,
            context.trajectories_source_order,
            result_values,
        ):
            internal_correct = bool(trajectory.get("final_correct"))
            official_correct = bool(official_result)
            per_query.append(
                {
                    "source_index": expected.source_index,
                    "query_id": expected.query_id,
                    "db_id": expected.db_id,
                    "difficulty": expected.difficulty,
                    "correct": official_correct,
                    "official_correct": official_correct,
                    "internal_correct": internal_correct,
                    "mismatch_class": CORRECTNESS_LABELS[
                        (internal_correct, official_correct)
                    ],
                    "internal_failure_type": trajectory.get(
                        "final_failure_type"
                    ),
                }
            )
        difficulty_result = module.compute_acc_by_diff(
            official_results, str(context.examples_path)
        )
        difficulty_scores = difficulty_result[:4]
        difficulty_counts = difficulty_result[4]

    labels = ("simple", "moderate", "challenging", "total")
    difficulty = {
        label: {
            "count": int(count),
            "accuracy": float(accuracy) / 100.0,
        }
        for label, count, accuracy in zip(
            labels, difficulty_counts, difficulty_scores
        )
    }
    database_manifest = _tree_layout_manifest(context.database_root)
    result = _base_result(
        context,
        final_correct_count=final_correct_count,
        evaluator=_official_provenance(
            context,
            repository=BIRD_REPOSITORY,
            commit=BIRD_COMMIT,
            source_hashes=source_hashes,
            options={
                "mode_predict": "gpt",
                "mode_gt": "gt",
                "data_mode": "dev",
                "num_cpus": num_cpus,
                "meta_time_out_seconds": timeout_seconds,
                "comparison": "set(predicted_rows) == set(gold_rows)",
            },
        ),
        inputs={
            "predictions_jsonl": {
                "path": str(context.run_directory / "predictions.jsonl"),
                "sha256": context.immutable_hashes["predictions.jsonl"],
            },
            "native_input_sha256": native_hashes,
            "gold": {
                "path": str(gold_path),
                "sha256": _sha256_file(gold_path),
            },
            "examples": {
                "path": str(context.examples_path),
                "sha256": _sha256_file(context.examples_path),
            },
            "tables": {
                "path": str(context.tables_path),
                "sha256": _sha256_file(context.tables_path),
            },
            "database": database_manifest,
        },
    )
    result["difficulty"] = difficulty
    result["correctness_comparison_counts"] = {
        label: sum(row["mismatch_class"] == label for row in per_query)
        for label in CORRECTNESS_LABELS.values()
    }
    result["per_query"] = per_query
    return result


def _single_line_spider_sql(sql: str, source_index: int) -> str:
    repository_path = str(REPOSITORY_ROOT)
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    from merit.evaluator import EvaluationError, render_spider_prediction

    try:
        return render_spider_prediction(sql, source_index)
    except EvaluationError as error:
        raise EvaluationContractError(str(error)) from error


def _capture_spider_scores(
    module: Any,
    *,
    gold_path: Path,
    prediction_path: Path,
    database_root: Path,
    etype: str,
    kmaps: Any,
    plug_value: bool,
    keep_distinct: bool,
) -> tuple[Mapping[str, Any], tuple[int, ...]]:
    captured: dict[str, Any] = {}
    execution_results: list[int] = []

    def capture(scores: Mapping[str, Any], _: str, **__: Any) -> None:
        captured["scores"] = copy.deepcopy(scores)

    original_print_scores = module.print_scores
    original_exec_match = module.eval_exec_match

    def capture_execution(*args: Any, **kwargs: Any) -> int:
        result = original_exec_match(*args, **kwargs)
        if isinstance(result, bool) or not isinstance(result, int) or result not in {0, 1}:
            raise EvaluationContractError(
                f"Spider official execution result is not binary: {result!r}"
            )
        execution_results.append(result)
        return result

    module.print_scores = capture
    module.eval_exec_match = capture_execution
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            module.evaluate(
                str(gold_path),
                str(prediction_path),
                str(database_root),
                etype,
                kmaps,
                plug_value,
                keep_distinct,
                False,
            )
    finally:
        module.print_scores = original_print_scores
        module.eval_exec_match = original_exec_match
    if "scores" not in captured:
        raise EvaluationContractError(
            f"Spider official evaluator did not return {etype} scores"
        )
    return captured["scores"], tuple(execution_results)


def _exact_count(scores: Mapping[str, Any], metric: str) -> tuple[int, int, float]:
    all_scores = _require_mapping(scores.get("all"), "Spider scores.all")
    total = _require_integer(all_scores.get("count"), "Spider scores.all.count")
    accuracy = float(all_scores.get(metric))
    raw_count = accuracy * total
    count = int(round(raw_count))
    if not math.isclose(raw_count, count, rel_tol=0.0, abs_tol=1e-8):
        raise EvaluationContractError(
            f"Spider {metric} score does not resolve to an exact count"
        )
    return count, total, accuracy


def _spider_level_scores(
    scores: Mapping[str, Any], metric: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for level in ("easy", "medium", "hard", "extra", "all"):
        values = _require_mapping(scores.get(level), f"Spider scores.{level}")
        result[level] = {
            "count": _require_integer(
                values.get("count"), f"Spider scores.{level}.count"
            ),
            "accuracy": float(values.get(metric)),
        }
    return result


def _spider_git_commit(directory: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _evaluate_spider(
    context: RunContext,
    *,
    gold_path: Path,
    test_suite_root: Path,
    archive_path: Path | None,
    random_seed: int,
) -> dict[str, Any]:
    _validate_gold(gold_path, context.expected_examples)
    if test_suite_root.resolve() != context.database_root.resolve():
        raise EvaluationContractError(
            "Spider official database root differs from the configured protocol root"
        )
    source_directory = VENDOR_ROOT / "spider"
    source_hashes = _verify_sources(source_directory, SPIDER_SOURCE_SHA256)
    observed_commit = _spider_git_commit(source_directory)
    if observed_commit != SPIDER_COMMIT:
        raise EvaluationContractError(
            f"Spider evaluator commit mismatch: {observed_commit}"
        )
    canonical_missing = [
        expected.db_id
        for expected in context.expected_examples
        if not (
            test_suite_root
            / expected.db_id
            / f"{expected.db_id}.sqlite"
        ).is_file()
    ]
    if canonical_missing:
        raise EvaluationContractError(
            "Spider test suite lacks canonical databases: "
            + ", ".join(sorted(set(canonical_missing)))
        )

    _prepare_vendor_python()
    spider_path = str(source_directory)
    if spider_path not in sys.path:
        sys.path.insert(0, spider_path)
    module = _load_module(
        "merit_pinned_spider_evaluation", source_directory / "evaluation.py"
    )
    with tempfile.TemporaryDirectory(prefix="merit-spider-official-") as temp_name:
        temporary = Path(temp_name)
        native_prediction = temporary / "predictions.sql"
        native_gold = temporary / "dev_gold.sql"
        rendered_predictions = [
            _single_line_spider_sql(
                str(prediction["predicted_sql"]), expected.source_index
            )
            for prediction, expected in zip(
                context.predictions_source_order,
                context.expected_examples,
            )
        ]
        native_prediction.write_text(
            "\n".join(rendered_predictions) + "\n", encoding="utf-8"
        )
        native_gold.write_bytes(gold_path.read_bytes())
        native_hashes = {
            "predictions": _sha256_file(native_prediction),
            "gold": _sha256_file(native_gold),
        }

        random.seed(random_seed)
        execution_scores, execution_results = _capture_spider_scores(
            module,
            gold_path=native_gold,
            prediction_path=native_prediction,
            database_root=test_suite_root,
            etype="exec",
            kmaps=None,
            plug_value=False,
            keep_distinct=True,
        )
        kmaps = module.build_foreign_key_map_from_json(str(context.tables_path))
        exact_scores, exact_execution_results = _capture_spider_scores(
            module,
            gold_path=native_gold,
            prediction_path=native_prediction,
            database_root=test_suite_root,
            etype="match",
            kmaps=kmaps,
            plug_value=False,
            keep_distinct=False,
        )
        final_correct_count, total, execution_accuracy = _exact_count(
            execution_scores, "exec"
        )
        exact_correct_count, exact_total, exact_accuracy = _exact_count(
            exact_scores, "exact"
        )
    if total != len(context.expected_examples) or exact_total != total:
        raise EvaluationContractError("Spider official score count is incomplete")
    if len(execution_results) != total or sum(execution_results) != final_correct_count:
        raise EvaluationContractError(
            "Spider per-query execution results do not match aggregate scores"
        )
    if exact_execution_results:
        raise EvaluationContractError(
            "Spider exact-match evaluation unexpectedly executed SQL"
        )
    per_query = []
    for expected, trajectory, official_result in zip(
        context.expected_examples,
        context.trajectories_source_order,
        execution_results,
    ):
        internal_correct = bool(trajectory.get("final_correct"))
        official_correct = bool(official_result)
        per_query.append(
            {
                "source_index": expected.source_index,
                "query_id": expected.query_id,
                "db_id": expected.db_id,
                "official_correct": official_correct,
                "internal_correct": internal_correct,
                "mismatch_class": CORRECTNESS_LABELS[
                    (internal_correct, official_correct)
                ],
                "internal_failure_type": trajectory.get("final_failure_type"),
            }
        )

    database_manifest = _tree_layout_manifest(test_suite_root)
    archive: dict[str, Any] | None = None
    if archive_path is not None:
        if not archive_path.is_file():
            raise EvaluationContractError(
                f"Spider test-suite archive does not exist: {archive_path}"
            )
        archive_hash = _sha256_file(archive_path)
        if (
            archive_path
            == (REPOSITORY_ROOT / "data/_downloads/spider_test_suite.zip").resolve()
            and archive_hash != SPIDER_ARCHIVE_SHA256
        ):
            raise EvaluationContractError(
                "default Spider test-suite archive SHA256 does not match the pin"
            )
        archive = {
            "path": str(archive_path),
            "sha256": archive_hash,
        }

    result = _base_result(
        context,
        final_correct_count=final_correct_count,
        evaluator=_official_provenance(
            context,
            repository=SPIDER_REPOSITORY,
            commit=SPIDER_COMMIT,
            source_hashes=source_hashes,
            options={
                "headline_etype": "exec",
                "plug_value": False,
                "keep_distinct": True,
                "progress_bar_for_each_datapoint": False,
                "python_random_seed": random_seed,
                "per_database_timeout_seconds": 60,
                "exact_match_reference": {
                    "etype": "match",
                    "disable_value": True,
                    "disable_distinct": True,
                },
            },
        ),
        inputs={
            "predictions_jsonl": {
                "path": str(context.run_directory / "predictions.jsonl"),
                "sha256": context.immutable_hashes["predictions.jsonl"],
            },
            "native_input_sha256": native_hashes,
            "gold": {
                "path": str(gold_path),
                "sha256": _sha256_file(gold_path),
            },
            "examples": {
                "path": str(context.examples_path),
                "sha256": _sha256_file(context.examples_path),
            },
            "tables": {
                "path": str(context.tables_path),
                "sha256": _sha256_file(context.tables_path),
            },
            "test_suite_database": database_manifest,
            "test_suite_archive": archive,
        },
    )
    result["execution_accuracy"] = execution_accuracy
    result["difficulty"] = _spider_level_scores(execution_scores, "exec")
    result["secondary_metrics"] = {
        "exact_match": {
            "correct_count": exact_correct_count,
            "total_count": exact_total,
            "accuracy": exact_accuracy,
            "difficulty": _spider_level_scores(exact_scores, "exact"),
        }
    }
    result["correctness_comparison_counts"] = {
        label: sum(row["mismatch_class"] == label for row in per_query)
        for label in CORRECTNESS_LABELS.values()
    }
    result["per_query"] = per_query
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", help="completed MERIT run directory")
    parser.add_argument(
        "--gold-path",
        help="official SQL<TAB>db_id gold file; inferred from the dataset by default",
    )
    parser.add_argument(
        "--spider-test-suite-root",
        default=str(REPOSITORY_ROOT / "data/spider_test_suite/database"),
        help="root containing official Spider test-suite db_id directories",
    )
    parser.add_argument(
        "--spider-test-suite-archive",
        default=str(REPOSITORY_ROOT / "data/_downloads/spider_test_suite.zip"),
        help="archive retained for Spider test-suite provenance",
    )
    parser.add_argument("--spider-random-seed", type=int, default=0)
    parser.add_argument("--bird-num-cpus", type=int, default=1)
    parser.add_argument("--bird-timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = _load_completed_run(Path(args.run_directory))
    explicit_gold = (
        Path(args.gold_path).resolve() if args.gold_path is not None else None
    )
    if context.dataset_name == "bird":
        gold_path = explicit_gold or _bird_default_gold(context)
        result = _evaluate_bird(
            context,
            gold_path=gold_path.resolve(),
            num_cpus=args.bird_num_cpus,
            timeout_seconds=args.bird_timeout_seconds,
        )
    else:
        gold_path = explicit_gold or context.examples_path.parent / "dev_gold.sql"
        archive_path = (
            Path(args.spider_test_suite_archive).resolve()
            if args.spider_test_suite_archive
            else None
        )
        result = _evaluate_spider(
            context,
            gold_path=gold_path.resolve(),
            test_suite_root=Path(args.spider_test_suite_root).resolve(),
            archive_path=archive_path,
            random_seed=args.spider_random_seed,
        )

    _assert_artifacts_unchanged(
        context.run_directory, context.immutable_hashes
    )
    output = context.run_directory / "official_eval.json"
    _atomic_json(output, result)
    _assert_artifacts_unchanged(
        context.run_directory, context.immutable_hashes
    )
    print(
        f"{context.dataset_name}: official={result['final_correct_count']}/"
        f"{result['total_count']} internal={result['internal_final_correct_count']}/"
        f"{result['total_count']} matches_internal={result['matches_internal']}"
    )
    print(f"Wrote {output}")
    return 0 if result["matches_internal"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationContractError as error:
        print(f"official evaluation error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
