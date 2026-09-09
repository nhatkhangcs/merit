"""Declarative method policies shared by both datasets and the single runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from .config import METHODS, ExperimentConfig


_MEMORY_POLARITIES = frozenset({"positive", "negative"})
HIGH_PRECISION_RETRIEVAL_TYPES = frozenset(
    {"Syntax", "Schema Linking", "Execution"}
)


@dataclass(frozen=True)
class ReflectionPolicy:
    """The complete Reflexion call, storage, and retrieval definition."""

    prompt_purpose: str = "reflection"
    generation_time: str = "after_each_failed_attempt"
    storage_time: str = "immediately_after_generation"
    retrieval_method: str = "local_episode_recency"
    calls_per_failed_attempt: int = 1


@dataclass(frozen=True)
class MethodPolicy:
    name: str
    report_label: str
    uses_shared_initial_cache: bool
    performs_repairs: bool
    uses_local_history: bool
    uses_current_feedback: bool
    static_example_source: str | None
    reflection: ReflectionPolicy | None
    uses_global_memory: bool
    updates_global_memory: bool
    memory_update_timing: str
    retrieval_strategy: str
    hard_type_filter: bool
    separate_polarity_pools: bool
    retrieved_polarities: tuple[str, ...]
    dense_rerank: bool
    bm25: bool
    random_selection: bool
    cross_database_only: bool
    causal_query_stream: bool
    exclude_current_query: bool
    stored_unit: str
    verification_policy: str
    report_as_online_learning: bool
    hard_filter_error_types: frozenset[str] | None = None

    def applies_hard_type_filter(self, error_type: str) -> bool:
        return self.hard_type_filter and (
            self.hard_filter_error_types is None
            or error_type in self.hard_filter_error_types
        )

    def validate(self) -> None:
        if self.name not in METHODS:
            raise ValueError(f"unknown method policy: {self.name}")
        if not self.uses_shared_initial_cache:
            raise ValueError(f"{self.name} must use the shared initial SQL cache")
        if self.reflection is not None:
            if self.reflection.calls_per_failed_attempt != 1:
                raise ValueError("Reflexion must count one reflection call per failed attempt")
            if self.name != "reflexion":
                raise ValueError("only reflexion may enable reflection generation")
        if self.static_example_source is not None and self.name != "vanilla":
            raise ValueError("fixed static examples belong only to vanilla")
        if not self.uses_global_memory and self.updates_global_memory:
            raise ValueError("a method cannot update disabled global memory")
        if not self.uses_global_memory and self.retrieval_strategy not in {
            "none",
            "fixed_static_examples",
            "local_reflection_recency",
        }:
            raise ValueError("global retrieval requires global memory")
        if self.separate_polarity_pools and not self.retrieved_polarities:
            raise ValueError("separate polarity retrieval requires an explicit polarity set")
        unsupported_polarities = set(self.retrieved_polarities) - _MEMORY_POLARITIES
        if unsupported_polarities:
            raise ValueError(
                f"unsupported retrieved polarities: {sorted(unsupported_polarities)}"
            )
        if self.random_selection and (self.dense_rerank or self.bm25):
            raise ValueError("random_same_type cannot also score with dense or BM25")
        if self.hard_filter_error_types is not None:
            if not self.hard_type_filter:
                raise ValueError("conditional type filtering requires a hard type filter")
            if not self.hard_filter_error_types:
                raise ValueError("conditional type filtering requires explicit error types")
        if self.causal_query_stream and self.uses_global_memory and not self.exclude_current_query:
            raise ValueError("causal global memory must exclude the current query")
        if self.name == "dynamic_rag":
            if self.hard_type_filter or self.separate_polarity_pools:
                raise ValueError("dynamic_rag is untyped and uses one memory pool")
        if self.name == "transductive_batch":
            if self.causal_query_stream or self.report_as_online_learning:
                raise ValueError("transductive_batch must never be labelled online learning")


_NO_MEMORY = MethodPolicy(
    name="iterative",
    report_label="iterative",
    uses_shared_initial_cache=True,
    performs_repairs=True,
    uses_local_history=True,
    uses_current_feedback=True,
    static_example_source=None,
    reflection=None,
    uses_global_memory=False,
    updates_global_memory=False,
    memory_update_timing="never",
    retrieval_strategy="none",
    hard_type_filter=False,
    separate_polarity_pools=False,
    retrieved_polarities=(),
    dense_rerank=False,
    bm25=False,
    random_selection=False,
    cross_database_only=False,
    causal_query_stream=True,
    exclude_current_query=True,
    stored_unit="none",
    verification_policy="none",
    report_as_online_learning=False,
)

_MERIT_FULL = MethodPolicy(
    name="merit_full",
    report_label="merit_full",
    uses_shared_initial_cache=True,
    performs_repairs=True,
    uses_local_history=True,
    uses_current_feedback=True,
    static_example_source=None,
    reflection=None,
    uses_global_memory=True,
    updates_global_memory=True,
    memory_update_timing="after_episode_finalization",
    retrieval_strategy="typed_hybrid",
    hard_type_filter=True,
    separate_polarity_pools=True,
    retrieved_polarities=("positive", "negative"),
    dense_rerank=True,
    bm25=True,
    random_selection=False,
    cross_database_only=False,
    causal_query_stream=True,
    exclude_current_query=True,
    stored_unit="typed_repair_evidence",
    verification_policy=(
        "positive_only_when_oracle_confirmed;"
        "at_most_one_observed_negative_after_unsuccessful_episode"
    ),
    report_as_online_learning=True,
)


_POLICIES = {
    "zeroshot": replace(
        _NO_MEMORY,
        name="zeroshot",
        report_label="zeroshot",
        performs_repairs=False,
        uses_local_history=False,
        uses_current_feedback=False,
    ),
    "iterative": _NO_MEMORY,
    "vanilla": replace(
        _NO_MEMORY,
        name="vanilla",
        report_label="vanilla",
        static_example_source="config.static_examples_path",
        retrieval_strategy="fixed_static_examples",
        stored_unit="fixed_static_example_with_source",
    ),
    "reflexion": replace(
        _NO_MEMORY,
        name="reflexion",
        report_label="reflexion",
        reflection=ReflectionPolicy(),
        retrieval_strategy="local_reflection_recency",
        stored_unit="local_reflection_text",
        verification_policy="reflection_uses_public_outcome_only",
    ),
    "dynamic_rag": replace(
        _MERIT_FULL,
        name="dynamic_rag",
        report_label="dynamic_rag",
        retrieval_strategy="untyped_hybrid_single_pool",
        hard_type_filter=False,
        separate_polarity_pools=False,
        retrieved_polarities=("positive",),
        stored_unit="question_schema_attempt_outcome",
        verification_policy="store_only_evaluator_confirmed_successes",
    ),
    "merit": replace(_MERIT_FULL, name="merit", report_label="merit"),
    "merit_full": _MERIT_FULL,
    "confidence_aware_type_filter": replace(
        _MERIT_FULL,
        name="confidence_aware_type_filter",
        report_label="Confidence-aware type filter",
        retrieval_strategy="confidence_aware_typed_hybrid",
        hard_filter_error_types=HIGH_PRECISION_RETRIEVAL_TYPES,
    ),
    "positive_only": replace(
        _MERIT_FULL,
        name="positive_only",
        report_label="positive_only",
        retrieved_polarities=("positive",),
        verification_policy="positive_only_when_oracle_confirmed",
    ),
    "no_type_filter": replace(
        _MERIT_FULL,
        name="no_type_filter",
        report_label="no_type_filter",
        retrieval_strategy="untyped_hybrid_separate_polarity",
        hard_type_filter=False,
    ),
    "no_dense_rerank": replace(
        _MERIT_FULL,
        name="no_dense_rerank",
        report_label="no_dense_rerank",
        retrieval_strategy="typed_bm25_only",
        dense_rerank=False,
    ),
    "no_bm25": replace(
        _MERIT_FULL,
        name="no_bm25",
        report_label="no_bm25",
        retrieval_strategy="typed_dense_only",
        bm25=False,
    ),
    "random_same_type": replace(
        _MERIT_FULL,
        name="random_same_type",
        report_label="random_same_type",
        retrieval_strategy="random_same_type",
        dense_rerank=False,
        bm25=False,
        random_selection=True,
    ),
    "cross_database_only": replace(
        _MERIT_FULL,
        name="cross_database_only",
        report_label="cross_database_only",
        cross_database_only=True,
    ),
    "transductive_batch": replace(
        _MERIT_FULL,
        name="transductive_batch",
        report_label="transductive_batch (not online learning)",
        memory_update_timing="after_synchronous_batch_iteration",
        causal_query_stream=False,
        exclude_current_query=False,
        report_as_online_learning=False,
    ),
}

if set(_POLICIES) != set(METHODS):
    missing = sorted(set(METHODS) - set(_POLICIES))
    extra = sorted(set(_POLICIES) - set(METHODS))
    raise RuntimeError(f"method policy registry mismatch; missing={missing}, extra={extra}")

for _policy in _POLICIES.values():
    _policy.validate()

METHOD_POLICIES: Mapping[str, MethodPolicy] = MappingProxyType(_POLICIES)


def get_method_policy(method_name: str) -> MethodPolicy:
    try:
        return METHOD_POLICIES[method_name]
    except KeyError as exc:
        raise ValueError(f"unsupported method: {method_name}") from exc


def policy_for_config(config: ExperimentConfig) -> MethodPolicy:
    """Resolve and validate the policy-specific config requirements."""

    config.validate()
    policy = get_method_policy(config.method_name)
    if policy.static_example_source and not config.static_examples_path:
        raise ValueError("vanilla requires config.static_examples_path")
    return policy


__all__ = [
    "HIGH_PRECISION_RETRIEVAL_TYPES",
    "METHOD_POLICIES",
    "MethodPolicy",
    "ReflectionPolicy",
    "get_method_policy",
    "policy_for_config",
]
