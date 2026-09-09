"""Strict result validation and table export."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    canonical_json,
    config_from_dict,
    content_hash,
    evaluation_protocol_hash,
    evaluation_protocol_identity,
)
from .retrieval import CONFIDENCE_AWARE_VARIANT, validate_retrieval_log_alignment
from .metrics import (
    assert_shared_initial_predictions,
    derive_metrics,
    validate_official_counts,
)


REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "source_hash",
        "config_hash",
        "dataset_checksum",
        "database_manifest_hash",
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "package_versions",
        "evaluator_protocol",
        "evaluator_protocol_hash",
        "random_seed",
        "stream_order_seed",
        "feedback_regime",
        "method_name",
        "initial_cache_hash",
    }
)


REQUIRED_ARTIFACTS = frozenset(
    {
        "manifest.json",
        "config.json",
        "predictions.jsonl",
        "trajectories.jsonl",
        "retrieval_log.jsonl",
        "memory_positive.jsonl",
        "memory_negative.jsonl",
        "metrics.json",
        "official_eval.json",
        "stdout.log",
    }
)
MAIN_METHODS = frozenset(
    {
        "zeroshot",
        "iterative",
        "vanilla",
        "reflexion",
        "dynamic_rag",
        "merit",
        "merit_full",
        "positive_only",
        "no_type_filter",
        "no_dense_rerank",
        "no_bm25",
        "random_same_type",
        "cross_database_only",
        "confidence_aware_type_filter",
    }
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: malformed JSONL") from error
    return rows


@dataclass(frozen=True)
class ValidatedRun:
    directory: Path
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    trajectories: Sequence[Mapping[str, Any]]
    metrics: Mapping[str, Any]
    official: Mapping[str, Any]

    @property
    def dataset(self) -> str:
        return str(self.config["dataset"]["name"])

    @property
    def method(self) -> str:
        return str(self.manifest["method_name"])

    @property
    def stream_seed(self) -> int:
        return int(self.manifest["stream_order_seed"])


def _validate_manifest_config(
    directory: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        raise ValueError(f"{directory}: manifest is missing fields: {', '.join(missing)}")
    if manifest.get("completed") is not True:
        raise ValueError(f"{directory}: run manifest is not completed")
    observed_config_hash = content_hash(config)
    if manifest["config_hash"] != observed_config_hash:
        raise ValueError(
            f"{directory}: config hash differs from manifest; "
            f"manifest={manifest['config_hash']}, observed={observed_config_hash}"
        )

    try:
        retrieval = config["retrieval"]
        if not isinstance(retrieval, Mapping):
            raise TypeError("retrieval config must be an object")
        if (
            config.get("method_name") == CONFIDENCE_AWARE_VARIANT
            and "type_match_weight" not in retrieval
        ):
            raise ValueError(
                "confidence_aware_type_filter requires type_match_weight"
            )
        config_from_dict(config)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{directory}: persisted config is invalid: {error}"
        ) from error

    try:
        dataset_name = config["dataset"]["name"]
        protocol = config["evaluation_protocol"]
        timeout_seconds = config["reference_timeout_seconds"]
        protocol_hash = evaluation_protocol_hash(
            dataset_name, protocol, timeout_seconds
        )
        model = config["model"]
        expected = {
            "model_name": model["name"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "package_versions": config["package_versions"],
            "random_seed": config["random_seed"],
            "stream_order_seed": config["stream_order_seed"],
            "feedback_regime": config["feedback_regime"],
            "method_name": config["method_name"],
            "evaluator_protocol": protocol,
            "evaluator_protocol_hash": protocol_hash,
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"{directory}: config is missing run-identity fields") from error
    mismatches = {
        field: {"manifest": manifest[field], "config": expected_value}
        for field, expected_value in expected.items()
        if manifest[field] != expected_value
    }
    if mismatches:
        raise ValueError(
            f"{directory}: manifest/config identity differs: {canonical_json(mismatches)}"
        )


_STABLE_EVALUATOR_FIELDS = (
    "repository",
    "commit",
    "source_sha256",
    "options",
    "dependencies",
)


def _stable_evaluator_provenance(
    directory: Path, official: Mapping[str, Any]
) -> dict[str, Any]:
    evaluator = official.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError(f"{directory}: official evaluator provenance is missing")
    missing = [field for field in _STABLE_EVALUATOR_FIELDS if field not in evaluator]
    if missing:
        raise ValueError(
            f"{directory}: official evaluator provenance is missing fields: "
            + ", ".join(missing)
        )
    return {field: evaluator[field] for field in _STABLE_EVALUATOR_FIELDS}


def _validate_evaluation_protocol(
    directory: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    official: Mapping[str, Any],
) -> None:
    observed_provenance = _stable_evaluator_provenance(directory, official)
    evaluator = official["evaluator"]
    expected_hash = manifest["evaluator_protocol_hash"]
    if evaluator.get("protocol_hash") != expected_hash:
        raise ValueError(
            f"{directory}: official evaluator protocol hash differs from manifest"
        )
    identity = evaluation_protocol_identity(
        config["dataset"]["name"],
        config["evaluation_protocol"],
        config["reference_timeout_seconds"],
    )
    expected_provenance = {
        field: identity[field] for field in _STABLE_EVALUATOR_FIELDS
    }
    if observed_provenance != expected_provenance:
        raise ValueError(
            f"{directory}: official evaluator provenance differs from protocol identity"
        )


def _validate_prediction_identity(
    directory: Path,
    predictions: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
) -> None:
    if len(predictions) != len(trajectories):
        raise ValueError(f"{directory}: prediction/trajectory counts differ")
    seen: set[tuple[str, str]] = set()
    identity_fields = ("query_id", "db_id", "source_index", "stream_position")
    for row_number, (prediction, trajectory) in enumerate(
        zip(predictions, trajectories)
    ):
        missing_prediction = [field for field in identity_fields if field not in prediction]
        missing_trajectory = [field for field in identity_fields if field not in trajectory]
        if missing_prediction or missing_trajectory:
            raise ValueError(
                f"{directory}: prediction/trajectory identity fields are missing at "
                f"row {row_number}; prediction={missing_prediction}, "
                f"trajectory={missing_trajectory}"
            )
        prediction_identity = tuple(prediction[field] for field in identity_fields)
        trajectory_identity = tuple(trajectory[field] for field in identity_fields)
        if prediction_identity != trajectory_identity:
            raise ValueError(
                f"{directory}: prediction/trajectory identity differs at row {row_number}"
            )
        case_key = (str(prediction["db_id"]), str(prediction["query_id"]))
        if case_key in seen:
            raise ValueError(f"{directory}: duplicate prediction identity: {case_key}")
        seen.add(case_key)
        if int(prediction["stream_position"]) != row_number:
            raise ValueError(
                f"{directory}: stream position is not contiguous at row {row_number}"
            )
        if "predicted_sql" not in prediction:
            raise ValueError(f"{directory}: prediction row {row_number} has no predicted_sql")
        final_sql = trajectory.get("final_sql", trajectory["attempts"][-1])
        if final_sql != trajectory["attempts"][-1]:
            raise ValueError(
                f"{directory}: trajectory final_sql differs from final attempt at "
                f"row {row_number}"
            )
        if prediction["predicted_sql"] != final_sql:
            raise ValueError(
                f"{directory}: prediction SQL differs from trajectory at row {row_number}"
            )


def validate_run(run_directory: str | Path) -> ValidatedRun:
    directory = Path(run_directory)
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file())
    if missing:
        raise ValueError(f"{directory}: missing required artifacts: {', '.join(missing)}")

    manifest = _read_json(directory / "manifest.json")
    config = _read_json(directory / "config.json")
    if not isinstance(manifest, Mapping) or not isinstance(config, Mapping):
        raise ValueError(f"{directory}: manifest and config must be JSON objects")
    _validate_manifest_config(directory, manifest, config)
    trajectories = _read_jsonl(directory / "trajectories.jsonl")
    persisted_metrics = _read_json(directory / "metrics.json")
    official = _read_json(directory / "official_eval.json")
    if not isinstance(official, Mapping):
        raise ValueError(f"{directory}: official_eval.json must be a JSON object")
    _validate_evaluation_protocol(directory, manifest, config, official)
    derived = derive_metrics(trajectories)
    retrieval_logs = _read_jsonl(directory / "retrieval_log.jsonl")
    retrieval_config = config["retrieval"]
    validate_retrieval_log_alignment(
        retrieval_logs,
        trajectories,
        str(config["method_name"]),
        type_match_weight=float(retrieval_config.get("type_match_weight", 0.10)),
    )
    if persisted_metrics != derived:
        raise ValueError(f"{directory}: metrics.json is not trajectory-derived")
    validate_official_counts(derived, official)

    prediction_rows = _read_jsonl(directory / "predictions.jsonl")
    _validate_prediction_identity(directory, prediction_rows, trajectories)

    return ValidatedRun(directory, manifest, config, trajectories, derived, official)


def _control_key(run: ValidatedRun) -> dict[str, Any]:
    retrieval = run.config["retrieval"]
    return {
        "source_hash": run.manifest["source_hash"],
        "dataset_checksum": run.manifest["dataset_checksum"],
        "database_manifest_hash": run.manifest["database_manifest_hash"],
        "dataset_config": canonical_json(run.config["dataset"]),
        "model_config": canonical_json(run.config["model"]),
        "model_name": run.manifest["model_name"],
        "model_revision": run.manifest["model_revision"],
        "tokenizer_revision": run.manifest["tokenizer_revision"],
        "package_versions": canonical_json(run.manifest["package_versions"]),
        "generation": canonical_json(run.config["generation"]),
        "max_repair_steps": run.config["max_repair_steps"],
        "initial_cache_hash": run.manifest["initial_cache_hash"],
        "stream_order_seed": run.stream_seed,
        "random_seed": run.manifest["random_seed"],
        "feedback_regime": run.manifest["feedback_regime"],
        "retrieval_k": retrieval["max_positive"] + retrieval["max_negative"],
        "dense_weight": retrieval["dense_weight"],
        "bm25_weight": retrieval["bm25_weight"],
        "type_match_weight": retrieval.get("type_match_weight", 0.10),
        "evaluator_protocol_hash": run.manifest["evaluator_protocol_hash"],
        "official_evaluator_provenance": canonical_json(
            _stable_evaluator_provenance(run.directory, run.official)
        ),
    }


def validate_shared_controls(runs: Sequence[ValidatedRun]) -> None:
    grouped: dict[tuple[Any, ...], list[ValidatedRun]] = {}
    for run in runs:
        base = (run.dataset, run.stream_seed, run.manifest["feedback_regime"])
        grouped.setdefault(base, []).append(run)

    for group in grouped.values():
        compared = [run for run in group if run.method in MAIN_METHODS]
        if len(compared) < 2:
            continue
        reference = _control_key(compared[0])
        mismatched = {}
        for run in compared[1:]:
            candidate = _control_key(run)
            differences = sorted(
                field for field, value in candidate.items() if value != reference[field]
            )
            if differences:
                mismatched[run.method] = differences
        if mismatched:
            raise ValueError(
                "Shared controls differ for methods: " + canonical_json(mismatched)
            )
        assert_shared_initial_predictions(
            {run.method: run.trajectories for run in compared}
        )


def validate_stream_orders(runs: Sequence[ValidatedRun]) -> None:
    by_setting: dict[tuple[str, str, str], set[int]] = {}
    for run in runs:
        setting = (
            run.dataset,
            run.method,
            str(run.manifest["feedback_regime"]),
        )
        by_setting.setdefault(setting, set()).add(run.stream_seed)
    for (dataset, method, feedback_regime), seeds in by_setting.items():
        missing = {0, 1, 2}.difference(seeds)
        if missing:
            raise ValueError(
                f"{dataset}/{method}/{feedback_regime}: reported exports require "
                f"stream-order seeds 0, 1, and 2; missing {sorted(missing)}"
            )


def _summary_rows(runs: Iterable[ValidatedRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        metrics = run.metrics
        accounting = metrics["accounting"]
        rows.append(
            {
                "dataset": run.dataset,
                "method": run.method,
                "stream_seed": run.stream_seed,
                "feedback_regime": run.manifest["feedback_regime"],
                "initial_cache_hash": run.manifest["initial_cache_hash"],
                "n": metrics["total_examples"],
                "success_at_1": metrics["success_at_1"],
                "final_accuracy": metrics["final_accuracy"],
                "repaired_count": metrics["repaired_count"],
                "failed_count": metrics["failed_count"],
                "total_tokens": accounting["total_tokens"],
                "llm_calls": accounting["llm_calls"],
                "db_executions": accounting["db_executions"],
                "embedding_calls": accounting["embedding_calls"],
                "retrieval_calls": accounting["retrieval_calls"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            row["method"],
            row["feedback_regime"],
            row["stream_seed"],
        ),
    )


def export_results(
    run_directories: Sequence[str | Path],
    output_directory: str | Path,
    require_three_stream_orders: bool = True,
) -> list[dict[str, Any]]:
    runs = [validate_run(path) for path in run_directories]
    validate_shared_controls(runs)
    if require_three_stream_orders:
        validate_stream_orders(runs)
    rows = _summary_rows(runs)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown = []
    if rows:
        markdown.append("| " + " | ".join(fieldnames) + " |")
        markdown.append("| " + " | ".join("---" for _ in fieldnames) + " |")
        markdown.extend(
            "| " + " | ".join(str(row[field]) for field in fieldnames) + " |"
            for row in rows
        )
    (output / "results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return rows
