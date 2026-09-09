"""Gold-free dataset adapters and reproducibility metadata.

This module deliberately exposes only information available to generation and
retrieval. Reference SQL is loaded exclusively by :mod:`merit.evaluator`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DatasetConfig, content_hash


class DatasetError(ValueError):
    """Raised when a dataset cannot be represented deterministically."""


@dataclass(frozen=True)
class QueryExample:
    """A model-visible, reference-free text-to-SQL example."""

    dataset_name: str
    query_id: str
    source_index: int
    db_id: str
    question: str
    schema: str
    database_path: str
    evidence: str = ""
    difficulty: str = "unknown"


@dataclass(frozen=True)
class InvalidExample:
    """A source record omitted because required public fields were invalid."""

    source_index: int
    query_id: str
    reason: str


@dataclass(frozen=True)
class DatabaseManifestEntry:
    """Content identity for one SQLite database file."""

    db_id: str
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetBundle(Sequence[QueryExample]):
    """Loaded public examples together with immutable input identities."""

    name: str
    examples: tuple[QueryExample, ...]
    invalid_examples: tuple[InvalidExample, ...]
    dataset_checksum: str
    database_manifest_hash: str
    database_manifest: tuple[DatabaseManifestEntry, ...]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int | slice) -> QueryExample | tuple[QueryExample, ...]:
        return self.examples[index]

    def __iter__(self) -> Iterator[QueryExample]:
        return iter(self.examples)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without interpreting its contents."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_query_id(
    dataset_name: str,
    record: Mapping[str, Any],
    source_index: int,
) -> str:
    """Return the source-stable identifier shared by dataset and evaluator."""

    if dataset_name not in {"spider", "bird"}:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    if record.get("question_id") is not None:
        return str(record["question_id"])
    return str(source_index)


def build_database_manifest(
    database_root: str | Path,
) -> tuple[tuple[DatabaseManifestEntry, ...], str]:
    """Hash every SQLite file under ``database_root`` in relative-path order."""

    root = Path(database_root)
    if not root.is_dir():
        raise FileNotFoundError(f"database root does not exist: {root}")

    entries: list[DatabaseManifestEntry] = []
    for path in sorted(root.rglob("*.sqlite"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        db_id = relative.parts[0] if len(relative.parts) > 1 else path.stem
        entries.append(
            DatabaseManifestEntry(
                db_id=db_id,
                relative_path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )

    if not entries:
        raise DatasetError(f"no .sqlite databases found under {root}")

    frozen_entries = tuple(entries)
    manifest_hash = content_hash([entry.to_dict() for entry in frozen_entries])
    return frozen_entries, manifest_hash


def _load_json_list(path: str | Path, label: str) -> list[Mapping[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise DatasetError(f"{label} must contain a JSON list: {source}")
    if not all(isinstance(item, Mapping) for item in value):
        raise DatasetError(f"{label} entries must be JSON objects: {source}")
    return value


def _flatten_primary_keys(value: Any) -> set[int]:
    """Recursively flatten BIRD composite/nested primary-key declarations."""

    result: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, bool):
            raise DatasetError("boolean primary-key index is invalid")
        if isinstance(item, int):
            result.add(item)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        raise DatasetError(f"unsupported primary-key entry: {item!r}")

    visit(value)
    return result


def render_schema(record: Mapping[str, Any], *, nested_primary_keys: bool) -> str:
    """Render Spider/BIRD schema metadata using exact original names."""

    try:
        db_id = str(record["db_id"])
        table_names = list(record["table_names_original"])
        column_names = list(record["column_names_original"])
        column_types = list(record["column_types"])
    except (KeyError, TypeError) as exc:
        raise DatasetError("schema is missing required original-name fields") from exc

    raw_primary_keys = record.get("primary_keys", [])
    if not isinstance(raw_primary_keys, list):
        raise DatasetError("primary_keys must be a list")
    if not nested_primary_keys and any(
        isinstance(item, (list, tuple)) for item in raw_primary_keys
    ):
        raise DatasetError("nested primary keys are supported only by BIRD")
    primary_keys = _flatten_primary_keys(raw_primary_keys)

    foreign_key_targets: dict[int, int] = {}
    for pair in record.get("foreign_keys", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise DatasetError(f"invalid foreign-key pair for {db_id}: {pair!r}")
        source_index, target_index = pair
        if not isinstance(source_index, int) or not isinstance(target_index, int):
            raise DatasetError(f"foreign-key indices must be integers for {db_id}")
        foreign_key_targets[source_index] = target_index

    tables: dict[int, list[tuple[int, str, str]]] = {
        index: [] for index in range(len(table_names))
    }
    for column_index, column_entry in enumerate(column_names):
        if not isinstance(column_entry, (list, tuple)) or len(column_entry) != 2:
            raise DatasetError(f"invalid column declaration for {db_id}: {column_entry!r}")
        table_index, column_name = column_entry
        if table_index == -1:
            continue
        if not isinstance(table_index, int) or table_index not in tables:
            raise DatasetError(f"column references invalid table index in {db_id}")
        column_type = (
            str(column_types[column_index])
            if column_index < len(column_types)
            else "text"
        )
        tables[table_index].append((column_index, str(column_name), column_type))

    lines = [f"Database: {db_id}", ""]
    for table_index, table_name in enumerate(table_names):
        lines.append(f"Table: {table_name}")
        for column_index, column_name, column_type in tables[table_index]:
            attributes = [column_type]
            if column_index in primary_keys:
                attributes.append("PRIMARY KEY")
            target_index = foreign_key_targets.get(column_index)
            if target_index is not None:
                if target_index < 0 or target_index >= len(column_names):
                    raise DatasetError(f"foreign key target out of range in {db_id}")
                target_table_index, target_column = column_names[target_index]
                if (
                    not isinstance(target_table_index, int)
                    or target_table_index < 0
                    or target_table_index >= len(table_names)
                ):
                    raise DatasetError(f"foreign key target table out of range in {db_id}")
                attributes.append(
                    f"FOREIGN KEY -> {table_names[target_table_index]}.{target_column}"
                )
            lines.append(f"  - {column_name} ({', '.join(attributes)})")
        lines.append("")
    return "\n".join(lines).strip()


def _schema_map(
    tables_path: str | Path,
    *,
    nested_primary_keys: bool,
) -> dict[str, str]:
    schemas: dict[str, str] = {}
    for record in _load_json_list(tables_path, "tables file"):
        db_id = str(record.get("db_id", "")).strip()
        if not db_id:
            raise DatasetError("schema entry is missing db_id")
        if db_id in schemas:
            raise DatasetError(f"duplicate schema db_id: {db_id}")
        schemas[db_id] = render_schema(
            record,
            nested_primary_keys=nested_primary_keys,
        )
    return schemas


def _database_paths(
    root: Path,
    manifest: Sequence[DatabaseManifestEntry],
) -> dict[str, str]:
    by_db: dict[str, list[Path]] = {}
    for entry in manifest:
        by_db.setdefault(entry.db_id, []).append(root / entry.relative_path)

    resolved: dict[str, str] = {}
    for db_id, candidates in by_db.items():
        conventional = root / db_id / f"{db_id}.sqlite"
        if conventional in candidates:
            resolved[db_id] = str(conventional.resolve())
            continue
        if len(candidates) != 1:
            relative = [path.relative_to(root).as_posix() for path in candidates]
            raise DatasetError(f"ambiguous databases for {db_id}: {relative}")
        resolved[db_id] = str(candidates[0].resolve())
    return resolved


class SpiderDatasetAdapter:
    """Load public Spider fields while preserving raw source positions."""

    def __init__(self, config: DatasetConfig):
        if config.name != "spider":
            raise ValueError("SpiderDatasetAdapter requires dataset name 'spider'")
        self.config = config

    def load(self) -> DatasetBundle:
        return _load_bundle(self.config, nested_primary_keys=False)


class BirdDatasetAdapter:
    """Load public BIRD fields, including evidence and difficulty."""

    def __init__(self, config: DatasetConfig):
        if config.name != "bird":
            raise ValueError("BirdDatasetAdapter requires dataset name 'bird'")
        self.config = config

    def load(self) -> DatasetBundle:
        return _load_bundle(self.config, nested_primary_keys=True)


def _load_bundle(
    config: DatasetConfig,
    *,
    nested_primary_keys: bool,
) -> DatasetBundle:
    config.validate()
    schemas = _schema_map(
        config.tables_path,
        nested_primary_keys=nested_primary_keys,
    )
    manifest, manifest_hash = build_database_manifest(config.database_root)
    database_paths = _database_paths(Path(config.database_root), manifest)
    records = _load_json_list(config.examples_path, "examples file")

    examples: list[QueryExample] = []
    invalid: list[InvalidExample] = []
    seen_ids: set[str] = set()
    for source_index, record in enumerate(records):
        query_id = stable_query_id(config.name, record, source_index)
        if query_id in seen_ids:
            raise DatasetError(f"duplicate query_id: {query_id}")
        seen_ids.add(query_id)

        db_id = str(record.get("db_id", "")).strip()
        question = str(record.get("question", "")).strip()
        reason = ""
        if not db_id:
            reason = "missing db_id"
        elif not question:
            reason = "missing question"
        elif db_id not in schemas:
            reason = f"missing schema: {db_id}"
        elif db_id not in database_paths:
            reason = f"missing database: {db_id}"

        if reason:
            invalid.append(InvalidExample(source_index, query_id, reason))
            continue

        evidence = (
            str(record.get("evidence") or "").strip()
            if config.name == "bird"
            else ""
        )
        difficulty = (
            str(record.get("difficulty") or "unknown").strip()
            if config.name == "bird"
            else str(record.get("hardness") or "unknown").strip()
        )
        examples.append(
            QueryExample(
                dataset_name=config.name,
                query_id=query_id,
                source_index=source_index,
                db_id=db_id,
                question=question,
                schema=schemas[db_id],
                database_path=database_paths[db_id],
                evidence=evidence,
                difficulty=difficulty or "unknown",
            )
        )

    dataset_checksum = content_hash(
        {
            "dataset_name": config.name,
            "examples_sha256": sha256_file(config.examples_path),
            "tables_sha256": sha256_file(config.tables_path),
        }
    )
    return DatasetBundle(
        name=config.name,
        examples=tuple(examples),
        invalid_examples=tuple(invalid),
        dataset_checksum=dataset_checksum,
        database_manifest_hash=manifest_hash,
        database_manifest=manifest,
    )


def load_spider_dataset(config: DatasetConfig) -> DatasetBundle:
    return SpiderDatasetAdapter(config).load()


def load_bird_dataset(config: DatasetConfig) -> DatasetBundle:
    return BirdDatasetAdapter(config).load()


def load_dataset(config: DatasetConfig) -> DatasetBundle:
    """Dispatch to the configured gold-free dataset adapter."""

    config.validate()
    adapters = {
        "spider": SpiderDatasetAdapter,
        "bird": BirdDatasetAdapter,
    }
    return adapters[config.name](config).load()


__all__ = [
    "BirdDatasetAdapter",
    "DatabaseManifestEntry",
    "DatasetBundle",
    "DatasetError",
    "InvalidExample",
    "QueryExample",
    "SpiderDatasetAdapter",
    "build_database_manifest",
    "load_bird_dataset",
    "load_dataset",
    "load_spider_dataset",
    "render_schema",
    "sha256_file",
    "stable_query_id",
]
