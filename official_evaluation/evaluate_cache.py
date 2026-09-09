#!/usr/bin/env python3
"""Validate a v5 initial cache against the pinned official evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from merit.config import DatasetConfig, ExperimentConfig, load_config
from merit.dataset import DatasetBundle, QueryExample, load_dataset
from merit.generation import (
    InitialCacheIdentity,
    InitialSQLCache,
    InitialSQLCacheRecord,
    load_initial_cache,
)
from merit.prompts import build_initial_prompt
from official_evaluation import evaluate_run as official_run


VALIDATION_FORMAT_VERSION = "merit-initial-cache-official-validation-v1"


@dataclass(frozen=True)
class CacheValidationContext:
    config_path: Path
    cache_path: Path
    config: ExperimentConfig
    cache: InitialSQLCache
    records_source_order: tuple[InitialSQLCacheRecord, ...]
    official_context: official_run.RunContext
    config_sha256: str
    cache_file_sha256: str


def _resolved_dataset_config(config: ExperimentConfig) -> DatasetConfig:
    return DatasetConfig(
        name=config.dataset.name,
        examples_path=str(
            official_run._resolve_repository_path(config.dataset.examples_path)
        ),
        tables_path=str(
            official_run._resolve_repository_path(config.dataset.tables_path)
        ),
        database_root=str(
            official_run._resolve_repository_path(config.dataset.database_root)
        ),
    )


def _prompt_hashes(
    examples: Sequence[QueryExample],
) -> dict[tuple[str, str], str]:
    return {
        (example.query_id, example.db_id): build_initial_prompt(
            example.dataset_name,
            example.question,
            example.schema,
            example.evidence,
        ).hash
        for example in examples
    }


def _expected_examples(
    dataset: DatasetBundle,
    examples_path: Path,
) -> tuple[official_run.ExpectedExample, ...]:
    if dataset.invalid_examples:
        raise official_run.EvaluationContractError(
            "official cache validation requires zero invalid dataset examples"
        )
    source_examples = tuple(sorted(dataset, key=lambda example: example.source_index))
    if tuple(example.source_index for example in source_examples) != tuple(
        range(len(source_examples))
    ):
        raise official_run.EvaluationContractError(
            "dataset source_index coverage is not contiguous"
        )
    raw_examples = official_run._read_json(examples_path)
    if not isinstance(raw_examples, list) or not all(
        isinstance(record, Mapping) for record in raw_examples
    ):
        raise official_run.EvaluationContractError(
            "dataset examples must be a JSON list of objects"
        )
    if len(raw_examples) != len(source_examples):
        raise official_run.EvaluationContractError(
            "public dataset examples do not exactly cover source order"
        )

    expected: list[official_run.ExpectedExample] = []
    for source_index, (record, example) in enumerate(
        zip(raw_examples, source_examples)
    ):
        query_id = official_run._stable_query_id(
            dataset.name, record, source_index
        )
        db_id = str(record.get("db_id", "")).strip()
        observed_identity = (
            example.source_index,
            example.query_id,
            example.db_id,
        )
        if observed_identity != (source_index, query_id, db_id):
            raise official_run.EvaluationContractError(
                f"dataset source identity mismatch at index {source_index}"
            )
        reference_field = "query" if dataset.name == "spider" else "SQL"
        reference_sql = str(record.get(reference_field, "")).strip()
        if not reference_sql:
            raise official_run.EvaluationContractError(
                f"dataset reference SQL is empty at source index {source_index}"
            )
        difficulty_field = "hardness" if dataset.name == "spider" else "difficulty"
        expected.append(
            official_run.ExpectedExample(
                source_index=source_index,
                query_id=query_id,
                db_id=db_id,
                difficulty=str(
                    record.get(difficulty_field) or "unknown"
                ).strip()
                or "unknown",
                reference_sql=reference_sql,
            )
        )
    return tuple(expected)


def _load_cache_context(
    config_path: Path,
    cache_path: Path,
) -> CacheValidationContext:
    resolved_config_path = config_path.resolve()
    resolved_cache_path = cache_path.resolve()
    if not resolved_config_path.is_file():
        raise official_run.EvaluationContractError(
            f"config file does not exist: {resolved_config_path}"
        )
    if not resolved_cache_path.is_file():
        raise official_run.EvaluationContractError(
            f"cache file does not exist: {resolved_cache_path}"
        )

    config = load_config(resolved_config_path)
    configured_cache_path = official_run._resolve_repository_path(
        config.initial_cache_path
    )
    if configured_cache_path != resolved_cache_path:
        raise official_run.EvaluationContractError(
            "provided cache path differs from config.initial_cache_path"
        )
    resolved_config = replace(
        config,
        dataset=_resolved_dataset_config(config),
        initial_cache_path=str(resolved_cache_path),
    )
    dataset = load_dataset(resolved_config.dataset)
    expected_examples = _expected_examples(
        dataset, Path(resolved_config.dataset.examples_path)
    )
    prompt_hashes = _prompt_hashes(tuple(dataset))
    expected_identity = InitialCacheIdentity.from_config(
        resolved_config,
        dataset_checksum=dataset.dataset_checksum,
        database_manifest_hash=dataset.database_manifest_hash,
    )
    cache = load_initial_cache(
        resolved_cache_path,
        expected_identity=expected_identity,
        expected_prompt_hashes=prompt_hashes,
    )
    records_by_query = cache.by_query()
    source_examples = tuple(sorted(dataset, key=lambda example: example.source_index))
    records_source_order = tuple(
        records_by_query[(example.query_id, example.db_id)]
        for example in source_examples
    )
    empty = [
        expected.source_index
        for expected, record in zip(expected_examples, records_source_order)
        if not record.initial_sql.strip()
    ]
    if empty:
        raise official_run.EvaluationContractError(
            f"cache contains empty initial SQL at source indexes: {empty}"
        )

    configured_database = official_run._configured_database_manifest(
        Path(resolved_config.dataset.database_root)
    )
    if configured_database["sha256"] != dataset.database_manifest_hash:
        raise official_run.EvaluationContractError(
            "configured database manifest differs from the cache identity"
        )
    config_sha256 = official_run._sha256_file(resolved_config_path)
    cache_file_sha256 = official_run._sha256_file(resolved_cache_path)
    live_source_hash = official_run._live_source_hash()
    internal_correct_count = sum(
        record.oracle_correct for record in records_source_order
    )
    predictions = tuple(
        {
            "query_id": expected.query_id,
            "db_id": expected.db_id,
            "source_index": expected.source_index,
            "stream_position": expected.source_index,
            "predicted_sql": record.initial_sql,
        }
        for expected, record in zip(expected_examples, records_source_order)
    )
    trajectories = tuple(
        {
            "query_id": expected.query_id,
            "db_id": expected.db_id,
            "source_index": expected.source_index,
            "stream_position": expected.source_index,
            "final_sql": record.initial_sql,
            "final_correct": record.oracle_correct,
            "final_failure_type": None,
        }
        for expected, record in zip(expected_examples, records_source_order)
    )
    official_context = official_run.RunContext(
        run_directory=resolved_cache_path.parent,
        dataset_name=dataset.name,
        config=resolved_config.to_dict(),
        manifest={
            "run_id": f"initial-cache:{cache.cache_hash}",
            "source_hash": live_source_hash,
            "config_hash": official_run._canonical_hash(
                resolved_config.to_dict()
            ),
            "dataset_checksum": dataset.dataset_checksum,
            "database_manifest_hash": dataset.database_manifest_hash,
        },
        metrics={
            "total_examples": len(expected_examples),
            "final_correct_count": internal_correct_count,
        },
        examples_path=Path(resolved_config.dataset.examples_path),
        tables_path=Path(resolved_config.dataset.tables_path),
        database_root=Path(resolved_config.dataset.database_root),
        expected_examples=expected_examples,
        predictions_source_order=predictions,
        trajectories_source_order=trajectories,
        immutable_hashes={"predictions.jsonl": cache_file_sha256},
        live_source_hash=live_source_hash,
        configured_database_manifest=configured_database,
    )
    return CacheValidationContext(
        config_path=resolved_config_path,
        cache_path=resolved_cache_path,
        config=resolved_config,
        cache=cache,
        records_source_order=records_source_order,
        official_context=official_context,
        config_sha256=config_sha256,
        cache_file_sha256=cache_file_sha256,
    )


def _score_cache(context: CacheValidationContext) -> Mapping[str, Any]:
    official_context = context.official_context
    if official_context.dataset_name == "bird":
        return official_run._evaluate_bird(
            official_context,
            gold_path=(
                official_context.examples_path.parent / "dev.sql"
            ).resolve(),
            num_cpus=1,
            timeout_seconds=30.0,
        )
    return official_run._evaluate_spider(
        official_context,
        gold_path=(
            official_context.examples_path.parent / "dev_gold.sql"
        ).resolve(),
        test_suite_root=official_context.database_root.resolve(),
        archive_path=(
            REPOSITORY_ROOT / "data/_downloads/spider_test_suite.zip"
        ).resolve(),
        random_seed=0,
    )


def _validation_result(
    context: CacheValidationContext,
    official_result: Mapping[str, Any],
) -> dict[str, Any]:
    raw_per_query = official_result.get("per_query")
    if not isinstance(raw_per_query, list):
        raise official_run.EvaluationContractError(
            "official evaluator did not return per-query results"
        )
    expected_examples = context.official_context.expected_examples
    if len(raw_per_query) != len(expected_examples):
        raise official_run.EvaluationContractError(
            "official per-query result count differs from the cache"
        )
    per_query: list[dict[str, Any]] = []
    for expected, record, official_row in zip(
        expected_examples,
        context.records_source_order,
        raw_per_query,
    ):
        if not isinstance(official_row, Mapping):
            raise official_run.EvaluationContractError(
                "official per-query result is not an object"
            )
        observed_identity = (
            official_row.get("source_index"),
            str(official_row.get("query_id")),
            str(official_row.get("db_id")),
        )
        if observed_identity != (
            expected.source_index,
            expected.query_id,
            expected.db_id,
        ):
            raise official_run.EvaluationContractError(
                f"official result identity mismatch at {expected.source_index}"
            )
        official_correct = official_row.get("official_correct")
        if type(official_correct) is not bool:
            raise official_run.EvaluationContractError(
                f"official result is not boolean at {expected.source_index}"
            )
        internal_correct = record.oracle_correct
        per_query.append(
            {
                "source_index": expected.source_index,
                "query_id": expected.query_id,
                "db_id": expected.db_id,
                "difficulty": expected.difficulty,
                "cache_execution_status": record.execution_status,
                "internal_correct": internal_correct,
                "official_correct": official_correct,
                "matches": internal_correct is official_correct,
                "mismatch_class": official_run.CORRECTNESS_LABELS[
                    (internal_correct, official_correct)
                ],
            }
        )

    total_count = len(per_query)
    internal_correct_count = sum(row["internal_correct"] for row in per_query)
    official_correct_count = sum(row["official_correct"] for row in per_query)
    returned_count = official_result.get("final_correct_count")
    if (
        isinstance(returned_count, bool)
        or not isinstance(returned_count, int)
        or returned_count != official_correct_count
    ):
        raise official_run.EvaluationContractError(
            "official aggregate count differs from per-query results"
        )
    evaluator = official_result.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise official_run.EvaluationContractError(
            "official evaluator provenance is missing"
        )
    if (
        evaluator.get("protocol_hash")
        != context.cache.identity.evaluation_protocol_hash
    ):
        raise official_run.EvaluationContractError(
            "official evaluator protocol hash differs from the cache"
        )
    inputs = official_result.get("inputs")
    if not isinstance(inputs, Mapping):
        raise official_run.EvaluationContractError(
            "official evaluator input provenance is missing"
        )
    scorer_inputs = dict(inputs)
    scorer_inputs.pop("predictions_jsonl", None)
    scorer_inputs["cache"] = {
        "path": str(context.cache_path),
        "sha256": context.cache_file_sha256,
        "cache_hash": context.cache.cache_hash,
    }
    mismatch_count = sum(not row["matches"] for row in per_query)
    return {
        "format_version": VALIDATION_FORMAT_VERSION,
        "official": True,
        "dataset_name": context.cache.identity.dataset_name,
        "cache_hash": context.cache.cache_hash,
        "evaluation_protocol": context.cache.identity.evaluation_protocol,
        "protocol_hash": context.cache.identity.evaluation_protocol_hash,
        "total_count": total_count,
        "internal_correct_count": internal_correct_count,
        "official_correct_count": official_correct_count,
        "mismatch_count": mismatch_count,
        "matches_internal_vector": mismatch_count == 0,
        "correctness_comparison_counts": {
            label: sum(row["mismatch_class"] == label for row in per_query)
            for label in official_run.CORRECTNESS_LABELS.values()
        },
        "per_query": per_query,
        "evaluator": dict(evaluator),
        "inputs": scorer_inputs,
        "config": {
            "path": str(context.config_path),
            "sha256": context.config_sha256,
        },
        "validation_source_hash": context.official_context.live_source_hash,
        "evaluated_at": official_result.get("evaluated_at"),
    }


def _assert_inputs_unchanged(context: CacheValidationContext) -> None:
    if official_run._sha256_file(context.config_path) != context.config_sha256:
        raise official_run.EvaluationContractError(
            "config changed during official cache validation"
        )
    if (
        official_run._sha256_file(context.cache_path)
        != context.cache_file_sha256
    ):
        raise official_run.EvaluationContractError(
            "cache changed during official cache validation"
        )


def _atomic_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise official_run.EvaluationContractError(
            f"official cache validation already exists: {path}"
        )
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
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise official_run.EvaluationContractError(
                f"official cache validation already exists: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _validation_path(cache_path: Path) -> Path:
    return Path(str(cache_path.resolve()) + ".official_validation.json")


def run_validation(config_path: Path, cache_path: Path) -> int:
    output = _validation_path(cache_path)
    if output.exists():
        raise official_run.EvaluationContractError(
            f"official cache validation already exists: {output}"
        )
    context = _load_cache_context(config_path, cache_path)
    official_result = _score_cache(context)
    result = _validation_result(context, official_result)
    _assert_inputs_unchanged(context)
    _atomic_json_exclusive(output, result)
    try:
        _assert_inputs_unchanged(context)
    except official_run.EvaluationContractError:
        output.unlink()
        raise
    print(
        f"{result['dataset_name']}: official={result['official_correct_count']}/"
        f"{result['total_count']} internal={result['internal_correct_count']}/"
        f"{result['total_count']} vector_match="
        f"{result['matches_internal_vector']}"
    )
    print(f"Wrote {output}")
    return 0 if result["matches_internal_vector"] else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_validation(Path(args.config), Path(args.cache))
    except official_run.EvaluationContractError as error:
        print(f"official cache validation error: {error}", file=sys.stderr)
        return 1
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"official cache validation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
