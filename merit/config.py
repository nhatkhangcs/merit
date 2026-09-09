"""Validated, serializable experiment configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


FEEDBACK_REGIMES = frozenset({"denotation_confirmed", "dbms_only"})
DATASETS = frozenset({"spider", "bird"})
METHODS = frozenset(
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
        "transductive_batch",
    }
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)


EVALUATION_PROTOCOLS: Mapping[str, str] = {
    "spider": "spider_test_suite_exec_e97acc_v1",
    "bird": "bird_damo_exec_483554e_30s_v1",
}
EVALUATION_TIMEOUT_SECONDS: Mapping[str, float] = {
    "spider": 60.0,
    "bird": 30.0,
}
_OFFICIAL_EVALUATOR_DEPENDENCIES: Mapping[str, str] = {
    "click": "8.4.2",
    "func-timeout": "4.3.5",
    "joblib": "1.5.3",
    "nltk": "3.9.1",
    "regex": "2026.7.19",
    "sqlparse": "0.5.3",
    "tqdm": "4.67.0",
}
_EVALUATION_PROTOCOL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "spider": {
        "repository": "https://github.com/taoyds/test-suite-sql-eval",
        "commit": "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
        "source_sha256": {
            "evaluation.py": "7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779",
            "exec_eval.py": "29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7",
            "exec_subprocess.py": "1366694f8ad4d80cdd8fb45eb8c34f48f101fa88c3e0792bd3519f6bbe8530d9",
            "parse.py": "ef04211a6e1c1e142571157f5c1999613e3451084c044083b2de1977f1f622c5",
            "process_sql.py": "927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1",
        },
        "options": {
            "headline_etype": "exec",
            "plug_value": False,
            "keep_distinct": True,
            "progress_bar_for_each_datapoint": False,
            "python_random_seed": 0,
            "per_database_timeout_seconds": 60.0,
            "exact_match_reference": {
                "etype": "match",
                "disable_value": True,
                "disable_distinct": True,
            },
        },
    },
    "bird": {
        "repository": "https://github.com/AlibabaResearch/DAMO-ConvAI",
        "commit": "483554eae102996f5ec1f4feab4e78ef29c2a394",
        "source_sha256": {
            "evaluation.py": "2f591e559dc2d97e5b35d5b656e80b0c2edf968f0bb5a78ddfd1d88b4bbbc472",
            "run_evaluation.sh": "b3b3dc9daa06549b4818c38a320fde32d08fe8fc0fbac46d244267322a6cfbc5",
        },
        "options": {
            "num_cpus": 1,
            "mode_predict": "gpt",
            "mode_gt": "gt",
            "data_mode": "dev",
            "meta_time_out_seconds": 30.0,
            "comparison": "set(predicted_rows) == set(gold_rows)",
        },
    },
}

def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by every content hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evaluation_protocol_identity(
    dataset_name: str,
    protocol: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return the stable, canonical identity of an official evaluation protocol."""

    if dataset_name not in EVALUATION_PROTOCOLS:
        raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")
    expected_protocol = EVALUATION_PROTOCOLS[dataset_name]
    if protocol != expected_protocol:
        raise ValueError(
            f"evaluation_protocol for {dataset_name} must be {expected_protocol}"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("evaluation timeout must be a positive finite number")
    expected_timeout = EVALUATION_TIMEOUT_SECONDS[dataset_name]
    if float(timeout_seconds) != expected_timeout:
        raise ValueError(
            f"reference_timeout_seconds for {dataset_name} must be exactly "
            f"{expected_timeout}"
        )
    spec = _EVALUATION_PROTOCOL_SPECS[dataset_name]
    return {
        "dataset_name": dataset_name,
        "protocol": protocol,
        "timeout_seconds": expected_timeout,
        "repository": spec["repository"],
        "commit": spec["commit"],
        "source_sha256": dict(spec["source_sha256"]),
        "options": deepcopy(spec["options"]),
        "dependencies": dict(_OFFICIAL_EVALUATOR_DEPENDENCIES),
    }


def evaluation_protocol_hash(
    dataset_name: str,
    protocol: str,
    timeout_seconds: float,
) -> str:
    """Hash the complete stable identity of an official evaluation protocol."""

    return content_hash(
        evaluation_protocol_identity(dataset_name, protocol, timeout_seconds)
    )


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "a09a35458c702b33eeacc393d103063234e8bc28"
    tokenizer_revision: str = "a09a35458c702b33eeacc393d103063234e8bc28"
    embedding_name: str = "BAAI/bge-large-en-v1.5"
    embedding_revision: str = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    device_map: str = "auto"
    quantization: str = "4bit"

    def validate(self) -> None:
        revisions = {
            "model revision": self.revision,
            "tokenizer revision": self.tokenizer_revision,
            "embedding revision": self.embedding_revision,
        }
        unpinned = [label for label, value in revisions.items() if not _COMMIT_RE.fullmatch(value)]
        if unpinned:
            raise ValueError(f"Immutable commit hashes are required for: {', '.join(unpinned)}")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    examples_path: str
    tables_path: str
    database_root: str

    def validate(self) -> None:
        if self.name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {self.name}")
        missing = [
            label
            for label, value in (
                ("examples_path", self.examples_path),
                ("tables_path", self.tables_path),
                ("database_root", self.database_root),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Dataset paths must be explicit: {', '.join(missing)}")


@dataclass(frozen=True)
class RetrievalConfig:
    min_typed_pool: int = 3
    max_positive: int = 3
    max_negative: int = 1
    dense_weight: float = 0.75
    bm25_weight: float = 0.25
    type_match_weight: float = 0.10
    fallback_pool: str = "legal_same_polarity"
    index_backend: str = "faiss"

    def validate(self) -> None:
        if self.min_typed_pool < 1:
            raise ValueError("min_typed_pool must be positive")
        if self.max_positive < 0 or self.max_negative < 0:
            raise ValueError("retrieval limits cannot be negative")
        if self.fallback_pool != "legal_same_polarity":
            raise ValueError("fallback_pool must document the legal_same_polarity policy")
        if self.index_backend not in {"faiss", "python"}:
            raise ValueError("index_backend must be faiss or python")
        if (self.dense_weight, self.bm25_weight) != (0.75, 0.25):
            raise ValueError(
                "the protocol requires dense_weight=0.75 and bm25_weight=0.25"
            )
        if self.type_match_weight != 0.10:
            raise ValueError("the protocol requires type_match_weight=0.10")


@dataclass(frozen=True)
class GenerationConfig:
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    initial_do_sample: bool = False
    initial_temperature: float = 0.0
    repair_temperature: float = 0.0
    self_consistency_samples: int = 1

    def validate(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.max_input_tokens, self.max_output_tokens)
        ):
            raise ValueError("token limits must be positive integers")
        if not isinstance(self.initial_do_sample, bool):
            raise TypeError("initial_do_sample must be boolean")
        if self.initial_do_sample:
            raise ValueError("the main protocol requires greedy initial decoding")
        temperatures = (self.initial_temperature, self.repair_temperature)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in temperatures
        ):
            raise ValueError("generation temperatures must be finite numbers")
        if self.initial_temperature != 0:
            raise ValueError("the main greedy protocol requires initial_temperature=0")
        if self.repair_temperature < 0:
            raise ValueError("repair_temperature cannot be negative")
        if isinstance(self.self_consistency_samples, bool) or not isinstance(
            self.self_consistency_samples, int
        ):
            raise TypeError("self_consistency_samples must be an integer")
        if self.self_consistency_samples != 1:
            raise ValueError(
                "self-consistency is a separate ablation; the main protocol uses one initial SQL"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    run_root: str
    initial_cache_path: str
    evaluation_protocol: str
    method_name: str = "merit_full"
    feedback_regime: str = "denotation_confirmed"
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    random_seed: int = 42
    stream_order_seed: int = 0
    max_repair_steps: int = 7
    reference_timeout_seconds: float = 15.0
    static_examples_path: str = ""
    official_evaluator_command: tuple[str, ...] = ()
    package_versions: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.dataset.validate()
        self.model.validate()
        self.retrieval.validate()
        self.generation.validate()
        if self.method_name not in METHODS:
            raise ValueError(f"Unsupported method: {self.method_name}")
        if self.feedback_regime not in FEEDBACK_REGIMES:
            raise ValueError(f"Unsupported feedback regime: {self.feedback_regime}")
        if not self.run_root or not self.initial_cache_path:
            raise ValueError("run_root and initial_cache_path must be explicit")
        if self.max_repair_steps < 0:
            raise ValueError("max_repair_steps cannot be negative")
        evaluation_protocol_identity(
            self.dataset.name,
            self.evaluation_protocol,
            self.reference_timeout_seconds,
        )
        loose = [
            name
            for name, version in self.package_versions.items()
            if not version or any(op in version for op in (">", "<", "~", "*"))
        ]
        if loose:
            raise ValueError(f"Package versions must be exact pins: {', '.join(loose)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def evaluation_protocol_hash(self) -> str:
        return evaluation_protocol_hash(
            self.dataset.name,
            self.evaluation_protocol,
            self.reference_timeout_seconds,
        )


def _model_config(value: Mapping[str, Any] | None) -> ModelConfig:
    return ModelConfig(**dict(value or {}))


def _retrieval_config(value: Mapping[str, Any] | None) -> RetrievalConfig:
    return RetrievalConfig(**dict(value or {}))


def _generation_config(value: Mapping[str, Any] | None) -> GenerationConfig:
    return GenerationConfig(**dict(value or {}))


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    data = dict(raw)
    data["dataset"] = DatasetConfig(**dict(data["dataset"]))
    data["model"] = _model_config(data.get("model"))
    data["retrieval"] = _retrieval_config(data.get("retrieval"))
    data["generation"] = _generation_config(data.get("generation"))
    if "official_evaluator_command" in data:
        data["official_evaluator_command"] = tuple(data["official_evaluator_command"])
    config = ExperimentConfig(**data)
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
