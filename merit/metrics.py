"""Trajectory-derived experiment and classifier metrics."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


ACCOUNTING_FIELDS = (
    "initial_prompt_tokens",
    "initial_output_tokens",
    "repair_prompt_tokens",
    "repair_output_tokens",
    "total_prompt_tokens",
    "total_output_tokens",
    "total_tokens",
    "llm_calls",
    "db_executions",
    "embedding_calls",
    "retrieval_calls",
)


def _normalized_sql(sql: str) -> str:
    value = sql.strip().rstrip(";").lower()
    return re.sub(r"\s+", " ", value)


def _tokens(sql: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", _normalized_sql(sql))


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def diagnostic_components(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    oscillating = 0
    edit_by_step: dict[int, list[float]] = defaultdict(list)
    converged_steps: list[int] = []

    for trajectory in trajectories:
        attempts = [_normalized_sql(sql) for sql in trajectory["attempts"]]
        if any(sql in attempts[:index] for index, sql in enumerate(attempts[1:], start=1)):
            oscillating += 1
        if trajectory["final_correct"]:
            converged_steps.append(int(trajectory["repair_steps"]))
        for step, (before, after) in enumerate(zip(attempts, attempts[1:]), start=1):
            before_tokens = _tokens(before)
            after_tokens = _tokens(after)
            denominator = max(len(before_tokens), len(after_tokens), 1)
            edit_by_step[step].append(_edit_distance(before_tokens, after_tokens) / denominator)

    total = len(trajectories)
    return {
        "oscillation_rate": oscillating / total if total else 0.0,
        "convergence_variance": (
            statistics.pvariance(converged_steps) if len(converged_steps) > 1 else 0.0
        ),
        "mean_edit_distance_by_step": {
            str(step): sum(values) / len(values) for step, values in sorted(edit_by_step.items())
        },
    }


def derive_metrics(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive every aggregate from one-record-per-query trajectories."""

    query_ids: set[str] = set()
    for item in trajectories:
        query_id = str(item["query_id"])
        if query_id in query_ids:
            raise AssertionError(f"duplicate trajectory query_id: {query_id}")
        query_ids.add(query_id)

        if not isinstance(item["initial_correct"], bool):
            raise AssertionError(f"{query_id}: initial_correct must be boolean")
        if not isinstance(item["final_correct"], bool):
            raise AssertionError(f"{query_id}: final_correct must be boolean")
        if not isinstance(item["repair_steps"], int) or isinstance(
            item["repair_steps"], bool
        ):
            raise AssertionError(f"{query_id}: repair_steps must be an integer")

        attempts = item["attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise AssertionError(f"{query_id}: attempts must contain the initial SQL")
        expected_steps = len(attempts) - 1
        if item["repair_steps"] != expected_steps:
            raise AssertionError(
                f"{query_id}: repair_steps={item['repair_steps']} "
                f"but attempts imply {expected_steps}"
            )
        if len(item["outcomes"]) != len(attempts):
            raise AssertionError(f"{query_id}: outcomes must align one-to-one with attempts")
        if len(item["retrieved_entry_ids"]) != item["repair_steps"]:
            raise AssertionError(
                f"{query_id}: retrieved_entry_ids must align with repair generations"
            )
        if "final_sql" in item and item["final_sql"] != attempts[-1]:
            raise AssertionError(f"{query_id}: final_sql differs from the final attempt")

        prompt_tokens = item["prompt_tokens"]
        output_tokens = item["output_tokens"]
        accounting_row = item["accounting"]
        if set(accounting_row) != set(ACCOUNTING_FIELDS):
            missing = sorted(set(ACCOUNTING_FIELDS).difference(accounting_row))
            extra = sorted(set(accounting_row).difference(ACCOUNTING_FIELDS))
            raise AssertionError(
                f"{query_id}: accounting fields differ; missing={missing}, extra={extra}"
            )
        for field in ACCOUNTING_FIELDS:
            value = accounting_row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssertionError(
                    f"{query_id}: accounting.{field} must be a non-negative integer"
                )
        for label, values in (
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
        ):
            if not isinstance(values, list) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                raise AssertionError(
                    f"{query_id}: {label} must contain non-negative integers"
                )
        if len(prompt_tokens) != len(output_tokens):
            raise AssertionError(f"{query_id}: prompt/output token calls differ")
        if len(prompt_tokens) != accounting_row["llm_calls"]:
            raise AssertionError(f"{query_id}: token records must align with llm_calls")
        if accounting_row["llm_calls"] < len(attempts):
            raise AssertionError(f"{query_id}: SQL generations are missing from llm_calls")
        if prompt_tokens[0] != accounting_row["initial_prompt_tokens"]:
            raise AssertionError(f"{query_id}: initial prompt token count differs")
        if output_tokens[0] != accounting_row["initial_output_tokens"]:
            raise AssertionError(f"{query_id}: initial output token count differs")
        if sum(prompt_tokens[1:]) != accounting_row["repair_prompt_tokens"]:
            raise AssertionError(f"{query_id}: repair prompt token count differs")
        if sum(output_tokens[1:]) != accounting_row["repair_output_tokens"]:
            raise AssertionError(f"{query_id}: repair output token count differs")
        if accounting_row["total_prompt_tokens"] != (
            accounting_row["initial_prompt_tokens"]
            + accounting_row["repair_prompt_tokens"]
        ):
            raise AssertionError(f"{query_id}: prompt token accounting is inconsistent")
        if accounting_row["total_output_tokens"] != (
            accounting_row["initial_output_tokens"]
            + accounting_row["repair_output_tokens"]
        ):
            raise AssertionError(f"{query_id}: output token accounting is inconsistent")
        if accounting_row["total_tokens"] != (
            accounting_row["total_prompt_tokens"]
            + accounting_row["total_output_tokens"]
        ):
            raise AssertionError(f"{query_id}: total token accounting is inconsistent")
        if item["initial_correct"] and item["initial_failure_type"] is not None:
            raise AssertionError(
                f"{query_id}: initially correct examples cannot have a failure type"
            )
        if not item["initial_correct"] and not item["initial_failure_type"]:
            raise AssertionError(
                f"{query_id}: initially failed examples require a failure type"
            )
        if item["final_correct"] and item["final_failure_type"] is not None:
            raise AssertionError(
                f"{query_id}: finally correct examples cannot have a failure type"
            )
        if not item["final_correct"] and not item["final_failure_type"]:
            raise AssertionError(
                f"{query_id}: finally failed examples require a failure type"
            )

    total = len(trajectories)
    initial_correct_count = sum(bool(item["initial_correct"]) for item in trajectories)
    final_correct_count = sum(bool(item["final_correct"]) for item in trajectories)
    repaired_count = sum(
        not item["initial_correct"] and item["final_correct"] for item in trajectories
    )
    failed_count = total - final_correct_count

    initial_failure_breakdown = Counter(
        item["initial_failure_type"]
        for item in trajectories
        if not item["initial_correct"]
    )
    final_failure_breakdown = Counter(
        item["final_failure_type"] for item in trajectories if not item["final_correct"]
    )

    accounting = {
        field: sum(item["accounting"][field] for item in trajectories)
        for field in ACCOUNTING_FIELDS
    }
    if accounting["total_prompt_tokens"] != (
        accounting["initial_prompt_tokens"] + accounting["repair_prompt_tokens"]
    ):
        raise AssertionError("prompt token accounting is inconsistent")
    if accounting["total_output_tokens"] != (
        accounting["initial_output_tokens"] + accounting["repair_output_tokens"]
    ):
        raise AssertionError("output token accounting is inconsistent")
    if accounting["total_tokens"] != (
        accounting["total_prompt_tokens"] + accounting["total_output_tokens"]
    ):
        raise AssertionError("total token accounting is inconsistent")

    assert final_correct_count == sum(bool(item["final_correct"]) for item in trajectories)
    assert initial_correct_count == sum(bool(item["initial_correct"]) for item in trajectories)
    assert repaired_count == sum(
        not item["initial_correct"] and item["final_correct"] for item in trajectories
    )
    assert failed_count == total - final_correct_count
    assert sum(initial_failure_breakdown.values()) == total - initial_correct_count

    return {
        "total_examples": total,
        "initial_correct_count": initial_correct_count,
        "final_correct_count": final_correct_count,
        "repaired_count": repaired_count,
        "failed_count": failed_count,
        "success_at_1": initial_correct_count / total if total else 0.0,
        "final_accuracy": final_correct_count / total if total else 0.0,
        "initial_failure_breakdown": dict(sorted(initial_failure_breakdown.items())),
        "final_failure_breakdown": dict(sorted(final_failure_breakdown.items())),
        "accounting": accounting,
        "diagnostics": diagnostic_components(trajectories),
    }


def assert_shared_initial_predictions(
    method_trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Assert identical initial SQL and Success@1 across compared methods."""

    if not method_trajectories:
        return
    methods = iter(method_trajectories.items())
    reference_name, reference_rows = next(methods)
    reference = {
        str(item["query_id"]): (item["attempts"][0], bool(item["initial_correct"]))
        for item in reference_rows
    }
    for method_name, rows in methods:
        candidate = {
            str(item["query_id"]): (item["attempts"][0], bool(item["initial_correct"]))
            for item in rows
        }
        if candidate != reference:
            raise AssertionError(
                f"Initial predictions or Success@1 differ: {reference_name} vs {method_name}"
            )


def validate_official_counts(
    internal_metrics: Mapping[str, Any], official_evaluation: Mapping[str, Any]
) -> None:
    if official_evaluation.get("official") is not True:
        raise ValueError("A real Spider/BIRD official evaluation is required for export")
    expected = internal_metrics["final_correct_count"]
    observed = official_evaluation["final_correct_count"]
    if (
        type(expected) is not int
        or type(observed) is not int
        or expected < 0
        or observed < 0
    ):
        raise ValueError(
            "internal and official correct counts must be non-negative integers"
        )
    if expected != observed:
        raise ValueError(
            f"Internal and official correct counts differ: internal={expected}, "
            f"official={observed}"
        )


def classifier_validation_metrics(
    annotations: Iterable[Mapping[str, Any]],
    initial_trajectories: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(annotations)
    required = {
        "query_id",
        "db_id",
        "predicted_sql",
        "error_message",
        "predicted_type",
        "gold_type",
        "annotator_id",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"annotation is missing fields: {sorted(missing)}")

    cases: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cases[(str(row["db_id"]), str(row["query_id"]))].append(row)
    case_count = len(cases)

    initial_state: dict[tuple[str, str], bool] = {}
    for row in initial_trajectories:
        missing = {"query_id", "db_id", "initial_correct"}.difference(row)
        if missing:
            raise ValueError(
                f"initial trajectory is missing fields: {sorted(missing)}"
            )
        key = (str(row["db_id"]), str(row["query_id"]))
        if key in initial_state:
            raise ValueError(f"duplicate initial trajectory identity: {key}")
        initial_correct = row["initial_correct"]
        if not isinstance(initial_correct, bool):
            raise ValueError("initial_correct must be boolean")
        initial_state[key] = initial_correct
    unverified = sorted(key for key in cases if initial_state.get(key) is not False)
    if unverified:
        raise ValueError(
            "classifier annotations must be verified initially failed cases; "
            f"unverified={unverified}"
        )

    if not 150 <= case_count <= 200:
        raise ValueError(
            "classifier validation requires 150-200 unique initially failed cases; "
            f"received {case_count}"
        )

    canonical_rows = []
    for case_key, case_rows in cases.items():
        invariant_fields = (
            "predicted_sql",
            "error_message",
            "predicted_type",
            "gold_type",
        )
        conflicts = [
            field
            for field in invariant_fields
            if len({str(row[field]) for row in case_rows}) != 1
        ]
        if conflicts:
            raise ValueError(
                f"annotation case {case_key} has conflicting fields: {conflicts}"
            )
        annotators = {str(row["annotator_id"]) for row in case_rows}
        if "" in annotators:
            raise ValueError(f"annotation case {case_key} has an empty annotator_id")
        canonical_rows.append(case_rows[0])

    double_annotated = sum(
        len({str(row["annotator_id"]) for row in case_rows}) >= 2
        for case_rows in cases.values()
    )
    if double_annotated < 50:
        raise ValueError("at least 50 cases must have two annotators")

    covered = [row for row in canonical_rows if row["predicted_type"] != "Unknown"]
    correct = sum(
        row["predicted_type"] == row["gold_type"] for row in canonical_rows
    )
    labels = sorted(
        {str(row["gold_type"]) for row in canonical_rows}
        | {str(row["predicted_type"]) for row in canonical_rows}
    )
    confusion = {
        actual: {
            predicted: sum(
                row["gold_type"] == actual and row["predicted_type"] == predicted
                for row in canonical_rows
            )
            for predicted in labels
        }
        for actual in labels
    }
    f1_values = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in labels if actual != label)
        false_negative = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0
        )

    return {
        "coverage": len(covered) / case_count,
        "accuracy": correct / case_count,
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": confusion,
        "unknown_rate": 1 - len(covered) / case_count,
        "double_annotated_cases": double_annotated,
    }
