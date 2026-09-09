#!/usr/bin/env python3
"""Reannotate a canonical v4 initial cache under the configured v5 protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from merit.config import canonical_json, content_hash, load_config
from merit.dataset import (
    DatasetBundle,
    QueryExample,
    build_database_manifest,
    load_dataset,
)
from merit.evaluator import Evaluator
from merit.feedback import FeedbackRegime
from merit.generation import (
    CACHE_FORMAT_VERSION,
    InitialCacheIdentity,
    InitialEvaluation,
    InitialSQLCache,
    InitialSQLCacheRecord,
)
from merit.prompts import build_initial_prompt


LEGACY_CACHE_FORMAT_VERSION = "merit-initial-cache-v4"
_LEGACY_TOP_LEVEL_FIELDS = frozenset(
    {"format_version", "identity", "records", "cache_hash"}
)
_LEGACY_IDENTITY_FIELDS = frozenset(
    {
        "dataset_name",
        "dataset_checksum",
        "database_manifest_hash",
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "quantization",
        "prompt_format_version",
        "max_input_tokens",
        "max_output_tokens",
        "decoding_protocol",
        "annotation_regime",
    }
)
_GENERATION_IDENTITY_FIELDS = _LEGACY_IDENTITY_FIELDS - {
    "database_manifest_hash"
}
_RECORD_FIELDS = frozenset(
    {
        "query_id",
        "db_id",
        "prompt_hash",
        "initial_sql",
        "prompt_tokens",
        "output_tokens",
        "execution_status",
        "oracle_correct",
    }
)
_GENERATION_RECORD_FIELDS = (
    "query_id",
    "db_id",
    "prompt_hash",
    "initial_sql",
    "prompt_tokens",
    "output_tokens",
)
_PROTOCOL_IDENTITY_FIELDS = frozenset(
    {"evaluation_protocol", "evaluation_protocol_hash"}
)


@dataclass(frozen=True)
class LegacyInitialCache:
    identity: Mapping[str, Any]
    records: tuple[InitialSQLCacheRecord, ...]
    cache_hash: str


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    fields = set(value)
    if fields != expected:
        raise ValueError(
            f"{label} fields mismatch; "
            f"missing={sorted(expected - fields)}, extra={sorted(fields - expected)}"
        )


def _record_key(record: InitialSQLCacheRecord) -> tuple[str, str]:
    return record.query_id, record.db_id


def _canonical_records(
    records: tuple[InitialSQLCacheRecord, ...],
) -> tuple[InitialSQLCacheRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda record: canonical_json(
                [record.query_id, record.db_id]
            ),
        )
    )


def load_legacy_initial_cache(path: str | Path) -> LegacyInitialCache:
    """Load the one accepted legacy format and verify its canonical content hash."""

    raw = _require_object(
        json.loads(Path(path).read_text(encoding="utf-8")),
        "legacy initial cache",
    )
    _require_exact_fields(raw, _LEGACY_TOP_LEVEL_FIELDS, "legacy cache")
    if raw["format_version"] != LEGACY_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"expected {LEGACY_CACHE_FORMAT_VERSION}, "
            f"found {raw['format_version']!r}"
        )

    identity = _require_object(raw["identity"], "legacy cache identity")
    _require_exact_fields(
        identity,
        _LEGACY_IDENTITY_FIELDS,
        "legacy cache identity",
    )
    raw_records = raw["records"]
    if not isinstance(raw_records, list):
        raise TypeError("legacy cache records must be a JSON list")

    records: list[InitialSQLCacheRecord] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(raw_records):
        record_object = _require_object(
            value,
            f"legacy cache record {index}",
        )
        _require_exact_fields(
            record_object,
            _RECORD_FIELDS,
            f"legacy cache record {index}",
        )
        record = InitialSQLCacheRecord(**record_object)
        record.validate()
        key = _record_key(record)
        if key in seen:
            raise ValueError(
                f"duplicate legacy cache record: {canonical_json(key)}"
            )
        seen.add(key)
        records.append(record)

    canonical_records = _canonical_records(tuple(records))
    canonical_payload = {
        "format_version": LEGACY_CACHE_FORMAT_VERSION,
        "identity": dict(identity),
        "records": [asdict(record) for record in canonical_records],
    }
    expected_hash = content_hash(canonical_payload)
    if raw["cache_hash"] != expected_hash:
        raise ValueError("legacy initial cache content hash mismatch")
    return LegacyInitialCache(
        identity=dict(identity),
        records=canonical_records,
        cache_hash=expected_hash,
    )


def _legacy_identity(identity: InitialCacheIdentity) -> dict[str, Any]:
    value = asdict(identity)
    for field in _PROTOCOL_IDENTITY_FIELDS:
        del value[field]
    return value


def _generation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: value[field]
        for field in sorted(_GENERATION_IDENTITY_FIELDS)
    }


def _generation_projection(
    records: tuple[InitialSQLCacheRecord, ...],
) -> list[dict[str, Any]]:
    return [
        {
            field: getattr(record, field)
            for field in _GENERATION_RECORD_FIELDS
        }
        for record in _canonical_records(records)
    ]


def _prompt_hashes(
    dataset: DatasetBundle,
) -> dict[tuple[str, str], str]:
    return {
        (example.query_id, example.db_id): build_initial_prompt(
            example.dataset_name,
            example.question,
            example.schema,
            example.evidence,
        ).hash
        for example in dataset
    }


def _validate_source_binding(
    source: LegacyInitialCache,
    expected_identity: InitialCacheIdentity,
    prompt_hashes: Mapping[tuple[str, str], str],
) -> None:
    target_identity = _legacy_identity(expected_identity)
    if _generation_identity(source.identity) != _generation_identity(
        target_identity
    ):
        raise ValueError(
            "legacy cache generation identity does not match the configured "
            "dataset, model, tokenizer, prompt, or decoding protocol"
        )

    source_by_key = {
        _record_key(record): record
        for record in source.records
    }
    expected_keys = set(prompt_hashes)
    source_keys = set(source_by_key)
    if source_keys != expected_keys:
        raise ValueError(
            "legacy cache query coverage mismatch; "
            f"missing={sorted(expected_keys - source_keys)}, "
            f"extra={sorted(source_keys - expected_keys)}"
        )
    mismatched = sorted(
        key
        for key, expected_hash in prompt_hashes.items()
        if source_by_key[key].prompt_hash != expected_hash
    )
    if mismatched:
        raise ValueError(
            f"legacy cache prompt hash mismatch for queries: {mismatched}"
        )


def _database_manifest_attestation(
    source: LegacyInitialCache,
    expected_identity: InitialCacheIdentity,
    source_database_root: str | Path | None,
    target_database_root: str | Path,
) -> dict[str, Any]:
    source_hash = str(source.identity["database_manifest_hash"])
    target_hash = expected_identity.database_manifest_hash
    manifest_changed = source_hash != target_hash
    if manifest_changed and source_database_root is None:
        raise ValueError(
            "database manifest changed; --source-database-root is required "
            "to verify the legacy cache"
        )

    source_root = (
        Path(source_database_root).resolve()
        if source_database_root is not None
        else None
    )
    source_root_verified = False
    if source_root is not None:
        _, observed_hash = build_database_manifest(source_root)
        if observed_hash != source_hash:
            raise ValueError(
                "source database manifest hash does not match the legacy "
                f"cache: expected={source_hash}, observed={observed_hash}"
            )
        source_root_verified = True

    return {
        "source_database_manifest_hash": source_hash,
        "target_database_manifest_hash": target_hash,
        "database_manifest_changed": manifest_changed,
        "source_database_root": (
            str(source_root) if source_root is not None else None
        ),
        "target_database_root": str(Path(target_database_root).resolve()),
        "source_database_root_verified": source_root_verified,
    }


def _confirmed_evaluation(outcome: Any) -> InitialEvaluation:
    status = getattr(outcome.status, "value", outcome.status)
    evaluation = InitialEvaluation(
        execution_status=str(status),
        oracle_correct=outcome.oracle_correct,
    )
    evaluation.validate()
    return evaluation


def _reannotate_records(
    source: LegacyInitialCache,
    examples: Mapping[tuple[str, str], QueryExample],
    evaluator: Evaluator,
) -> tuple[InitialSQLCacheRecord, ...]:
    records: list[InitialSQLCacheRecord] = []
    for record in source.records:
        outcome = evaluator.evaluate(
            examples[_record_key(record)],
            record.initial_sql,
        )
        evaluation = _confirmed_evaluation(outcome)
        records.append(
            replace(
                record,
                execution_status=evaluation.execution_status,
                oracle_correct=evaluation.oracle_correct,
            )
        )
    return tuple(records)


def _label_counts(
    records: tuple[InitialSQLCacheRecord, ...],
) -> dict[str, int]:
    counts = Counter(
        f"{record.execution_status}:{str(record.oracle_correct).lower()}"
        for record in records
    )
    return dict(sorted(counts.items()))


def _transition_counts(
    source: tuple[InitialSQLCacheRecord, ...],
    target: tuple[InitialSQLCacheRecord, ...],
) -> dict[str, int]:
    counts = Counter(
        (
            f"{old.execution_status}:{str(old.oracle_correct).lower()}"
            f" -> {new.execution_status}:{str(new.oracle_correct).lower()}"
        )
        for old, new in zip(source, target, strict=True)
        if (
            old.execution_status,
            old.oracle_correct,
        )
        != (
            new.execution_status,
            new.oracle_correct,
        )
    )
    return dict(sorted(counts.items()))


def _migration_sidecar(
    source_path: Path,
    output_path: Path,
    source: LegacyInitialCache,
    target: InitialSQLCache,
    expected_identity: InitialCacheIdentity,
    database_manifest_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    source_projection = _generation_projection(source.records)
    target_projection = _generation_projection(target.records)
    if source_projection != target_projection:
        raise ValueError("reannotation changed generation fields")

    changed = [
        {"query_id": old.query_id, "db_id": old.db_id}
        for old, new in zip(source.records, target.records, strict=True)
        if (
            old.execution_status,
            old.oracle_correct,
        )
        != (
            new.execution_status,
            new.oracle_correct,
        )
    ]
    generation_hash = content_hash(source_projection)
    source_generation_identity = _generation_identity(source.identity)
    target_generation_identity = _generation_identity(
        _legacy_identity(expected_identity)
    )
    legacy_identity_hash = content_hash(source_generation_identity)
    target_generation_identity_hash = content_hash(
        target_generation_identity
    )
    if legacy_identity_hash != target_generation_identity_hash:
        raise ValueError("reannotation changed the shared generation identity")

    return {
        "migration": "merit-initial-cache-v4-to-v5-reannotation-v2",
        "source_cache_path": str(source_path),
        "output_cache_path": str(output_path),
        "source_format_version": LEGACY_CACHE_FORMAT_VERSION,
        "output_format_version": CACHE_FORMAT_VERSION,
        "source_cache_hash": source.cache_hash,
        "output_cache_hash": target.cache_hash,
        **database_manifest_attestation,
        "record_count": len(source.records),
        "changed_label_count": len(changed),
        "changed_label_ids": changed,
        "changed_execution_status_count": sum(
            old.execution_status != new.execution_status
            for old, new in zip(source.records, target.records, strict=True)
        ),
        "changed_oracle_correct_count": sum(
            old.oracle_correct is not new.oracle_correct
            for old, new in zip(source.records, target.records, strict=True)
        ),
        "label_transition_counts": _transition_counts(
            source.records,
            target.records,
        ),
        "source_label_counts": _label_counts(source.records),
        "output_label_counts": _label_counts(target.records),
        "evaluation_protocol": expected_identity.evaluation_protocol,
        "evaluation_protocol_hash": expected_identity.evaluation_protocol_hash,
        "generation_identity_attestation": {
            "fields": sorted(_GENERATION_IDENTITY_FIELDS),
            "source_hash": legacy_identity_hash,
            "output_hash": target_generation_identity_hash,
            "identical": True,
        },
        "generation_field_attestation": {
            "fields": list(_GENERATION_RECORD_FIELDS),
            "source_hash": generation_hash,
            "output_hash": content_hash(target_projection),
            "identical": True,
        },
        "generation_activity": {
            "backend_loaded": False,
            "model_loaded": False,
            "generation_calls": 0,
        },
    }


def migration_sidecar_path(output_path: str | Path) -> Path:
    return Path(f"{Path(output_path)}.migration.json")


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _stage_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _commit_new_pair(
    output_path: Path,
    cache_payload: Mapping[str, Any],
    sidecar_path: Path,
    sidecar_payload: Mapping[str, Any],
) -> None:
    existing = [
        str(path)
        for path in (output_path, sidecar_path)
        if _path_exists(path)
    ]
    if existing:
        raise FileExistsError(
            f"migration refuses to overwrite existing targets: {existing}"
        )

    cache_temporary = _stage_json(output_path, cache_payload)
    sidecar_temporary = _stage_json(sidecar_path, sidecar_payload)
    cache_committed = False
    sidecar_committed = False
    try:
        os.link(cache_temporary, output_path)
        cache_committed = True
        os.link(sidecar_temporary, sidecar_path)
        sidecar_committed = True
    finally:
        cache_temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
        if cache_committed and not sidecar_committed:
            output_path.unlink()


def reannotate_initial_cache(
    config_path: str | Path,
    source_cache_path: str | Path,
    output_path: str | Path | None = None,
    source_database_root: str | Path | None = None,
) -> tuple[InitialSQLCache, Path, Path]:
    config = load_config(config_path)
    source_path = Path(source_cache_path)
    destination = Path(
        config.initial_cache_path if output_path is None else output_path
    )
    if source_path.resolve() == destination.resolve():
        raise ValueError("migration output must differ from the v4 source cache")
    sidecar_path = migration_sidecar_path(destination)
    existing = [
        str(path)
        for path in (destination, sidecar_path)
        if _path_exists(path)
    ]
    if existing:
        raise FileExistsError(
            f"migration refuses to overwrite existing targets: {existing}"
        )

    source = load_legacy_initial_cache(source_path)
    dataset = load_dataset(config.dataset)
    expected_identity = InitialCacheIdentity.from_config(
        config,
        dataset_checksum=dataset.dataset_checksum,
        database_manifest_hash=dataset.database_manifest_hash,
    )
    expected_identity.validate()
    prompt_hashes = _prompt_hashes(dataset)
    _validate_source_binding(
        source,
        expected_identity,
        prompt_hashes,
    )
    database_manifest_attestation = _database_manifest_attestation(
        source,
        expected_identity,
        source_database_root,
        config.dataset.database_root,
    )

    examples = {
        (example.query_id, example.db_id): example
        for example in dataset
    }
    evaluator = Evaluator(
        config.dataset,
        FeedbackRegime.DENOTATION_CONFIRMED.value,
        evaluation_protocol=config.evaluation_protocol,
        reference_timeout_seconds=config.reference_timeout_seconds,
    )
    reannotated_records = _reannotate_records(
        source,
        examples,
        evaluator,
    )
    target = InitialSQLCache.create(
        expected_identity,
        reannotated_records,
    )
    sidecar = _migration_sidecar(
        source_path,
        destination,
        source,
        target,
        expected_identity,
        database_manifest_attestation,
    )
    _commit_new_pair(
        destination,
        target.to_dict(),
        sidecar_path,
        sidecar,
    )
    return target, destination, sidecar_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-database-root",
        help="Legacy database root; required when its manifest differs "
        "from the target config",
    )
    parser.add_argument("--config", required=True, help="Spider or BIRD JSON config")
    parser.add_argument(
        "--source-cache",
        required=True,
        help="Canonical merit-initial-cache-v4 JSON file",
    )
    parser.add_argument(
        "--output",
        help="New v5 path; defaults to config.initial_cache_path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache, output_path, sidecar_path = reannotate_initial_cache(
        args.config,
        args.source_cache,
        args.output,
        args.source_database_root,
    )
    print(
        f"Reannotated {len(cache.records)} records: "
        f"{output_path} ({cache.cache_hash})"
    )
    print(f"Migration attestation: {sidecar_path}")


if __name__ == "__main__":
    main()
