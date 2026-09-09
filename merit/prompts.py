"""Versioned, gold-free prompt construction for Spider and BIRD."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import DATASETS
from .feedback import OutcomeStatus


PROMPT_FORMAT_VERSION = "merit-prompts-v3"
PROMPT_VERSIONS = {
    ("spider", "initial"): "spider-initial-v3",
    ("spider", "repair"): "spider-repair-v3",
    ("spider", "reflection"): "spider-reflection-v3",
    ("bird", "initial"): "bird-initial-v3",
    ("bird", "repair"): "bird-repair-v3",
    ("bird", "reflection"): "bird-reflection-v3",
}


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt with an immutable template identity."""

    dataset_name: str
    purpose: str
    version: str
    text: str

    @property
    def hash(self) -> str:
        return prompt_hash(self.text)


@dataclass(frozen=True)
class StaticExample:
    """A fixed, externally sourced example used by the vanilla baseline."""

    source: str
    question: str
    sql: str
    evidence: str = ""

    def validate(self) -> None:
        if not self.source.strip():
            raise ValueError("static examples must record their source")
        if not self.question.strip() or not self.sql.strip():
            raise ValueError("static examples require a question and SQL")


def prompt_hash(prompt: str) -> str:
    """Hash the versioned UTF-8 application prompt before pinned chat framing."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _validate_dataset(dataset_name: str) -> str:
    normalized = dataset_name.lower().strip()
    if normalized not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return normalized


def prompt_version(dataset_name: str, purpose: str) -> str:
    dataset = _validate_dataset(dataset_name)
    try:
        return PROMPT_VERSIONS[(dataset, purpose)]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt purpose: {purpose}") from exc


def _bird_context(evidence: str) -> str:
    evidence_text = evidence.strip()
    if evidence_text:
        return (
            "EXTERNAL KNOWLEDGE / EVIDENCE:\n"
            f"{evidence_text}\n\n"
            "BIRD RULES:\n"
            "- Implement the evidence formula or computation exactly.\n"
            "- Wrap column names containing spaces or special characters in backticks, "
            "for example `Column Name`.\n"
        )
    return (
        "BIRD RULES:\n"
        "- Wrap column names containing spaces or special characters in backticks, "
        "for example `Column Name`.\n"
    )


def _dataset_context(dataset_name: str, evidence: str) -> str:
    if dataset_name == "bird":
        return _bird_context(evidence)
    if evidence.strip():
        raise ValueError("Spider prompts do not accept BIRD evidence")
    return ""


def _format_static_examples(examples: Sequence[StaticExample]) -> str:
    if not examples:
        return ""
    rendered = ["FIXED STATIC EXAMPLES:"]
    for index, example in enumerate(examples, start=1):
        example.validate()
        rendered.extend(
            [
                f"[Static example {index}; source={example.source}]",
                f"Question: {example.question}",
                f"Evidence: {example.evidence}" if example.evidence else "",
                f"SQL: {example.sql}",
            ]
        )
    return "\n".join(line for line in rendered if line) + "\n\n"


def build_initial_prompt(
    dataset_name: str,
    question: str,
    schema: str,
    evidence: str = "",
) -> Prompt:
    """Build the shared greedy initial prompt; it contains no method state."""

    dataset = _validate_dataset(dataset_name)
    version = prompt_version(dataset, "initial")
    context = _dataset_context(dataset, evidence)
    text = (
        f"PROMPT_VERSION: {version}\n"
        "You are an expert SQLite developer. Produce one SQL query for the question.\n\n"
        "DATABASE SCHEMA:\n"
        f"{schema.strip()}\n\n"
        f"{context}\n"
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "RULES:\n"
        "- Use only exact table and column names from the schema.\n"
        "- Do not create table or column aliases with AS.\n"
        "- Return exactly one query; do not provide alternatives.\n"
        "- Put the final SQL between <answer> and </answer> tags.\n"
    )
    return Prompt(dataset, "initial", version, text)


def _field(entry: Mapping[str, Any] | Any, name: str, default: str = "") -> str:
    if isinstance(entry, Mapping):
        value = entry.get(name, default)
    else:
        value = getattr(entry, name, default)
    return default if value is None else str(value)


def _format_positive_entries(entries: Sequence[Mapping[str, Any] | Any]) -> str:
    if not entries:
        return "(none)"
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"[Confirmed successful repair {index}]",
                f"Entry ID: {_field(entry, 'entry_id', _field(entry, 'case_id', 'unknown'))}",
                f"Error type: {_field(entry, 'error_type', 'Unknown')}",
                f"Failure context: {_field(entry, 'failure_context')}",
                f"Observed successful direction: "
                f"{_field(entry, 'successful_direction', _field(entry, 'correction_hint'))}",
                f"SQL delta: {_field(entry, 'attempted_sql_delta')}",
            ]
        )
    return "\n".join(lines)


def _format_negative_entries(entries: Sequence[Mapping[str, Any] | Any]) -> str:
    if not entries:
        return "(none)"
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"[OBSERVED FAILED DIRECTION {index}]",
                f"Entry ID: {_field(entry, 'entry_id', _field(entry, 'case_id', 'unknown'))}",
                f"Error type: {_field(entry, 'error_type', 'Unknown')}",
                f"Failure context: {_field(entry, 'failure_context')}",
                f"Attempted SQL delta: {_field(entry, 'attempted_sql_delta')}",
                f"Observed outcome: {_field(entry, 'observed_outcome')}",
                f"Observed DB error: {_field(entry, 'observed_db_error')}",
            ]
        )
    return "\n".join(lines)


def _format_attempts(attempts: Sequence[str]) -> str:
    if not attempts:
        raise ValueError("a repair prompt requires at least one prior attempt")
    return "\n\n".join(
        f"[Attempt {index} — observed unsuccessful]\n{sql}"
        for index, sql in enumerate(attempts, start=1)
    )


def _format_reflections(reflections: Sequence[str]) -> str:
    if not reflections:
        return "(none)"
    return "\n".join(
        f"[Local reflection {index}] {reflection}"
        for index, reflection in enumerate(reflections, start=1)
    )


def _repair_status(status: str | OutcomeStatus) -> str:
    raw_status = getattr(status, "value", status)
    try:
        normalized = OutcomeStatus(str(raw_status))
    except ValueError as exc:
        raise ValueError(f"unsupported repair status: {raw_status}") from exc
    if normalized is OutcomeStatus.CORRECT:
        raise ValueError("a CORRECT outcome must not receive a repair prompt")
    return normalized.value


def build_repair_prompt(
    dataset_name: str,
    question: str,
    schema: str,
    attempts: Sequence[str],
    *,
    status: str | OutcomeStatus,
    current_error_type: str,
    db_error: str | None = None,
    evidence: str = "",
    positive_entries: Sequence[Mapping[str, Any] | Any] = (),
    negative_entries: Sequence[Mapping[str, Any] | Any] = (),
    static_examples: Sequence[StaticExample] = (),
    reflections: Sequence[str] = (),
) -> Prompt:
    """Build a repair prompt from public feedback and legal retrieved entries."""

    dataset = _validate_dataset(dataset_name)
    version = prompt_version(dataset, "repair")
    context = _dataset_context(dataset, evidence)
    status_value = _repair_status(status)
    text = (
        f"PROMPT_VERSION: {version}\n"
        "You are an expert SQLite developer repairing an unsuccessful query.\n\n"
        f"{_format_static_examples(static_examples)}"
        "CONFIRMED SUCCESSFUL REPAIR DIRECTIONS:\n"
        f"{_format_positive_entries(positive_entries)}\n\n"
        "OBSERVED FAILED DIRECTIONS:\n"
        f"{_format_negative_entries(negative_entries)}\n\n"
        "LOCAL REFLECTIONS FROM THIS EPISODE:\n"
        f"{_format_reflections(reflections)}\n\n"
        "CURRENT FEEDBACK:\n"
        f"Status: {status_value}\n"
        f"Current error type: {current_error_type or 'Unknown'}\n"
        f"DB error: {db_error or '(none)'}\n\n"
        "DATABASE SCHEMA:\n"
        f"{schema.strip()}\n\n"
        f"{context}\n"
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "LOCAL ATTEMPT HISTORY:\n"
        f"{_format_attempts(attempts)}\n\n"
        "REPAIR RULES:\n"
        "- Produce a new SQL query rather than repeating a prior attempt.\n"
        "- Use only exact table and column names from the schema.\n"
        "- Treat failed directions only as observed evidence; do not invent a reason.\n"
        "- Put exactly one final SQL query between <answer> and </answer> tags.\n"
    )
    return Prompt(dataset, "repair", version, text)


def _outcome_text(outcome: Mapping[str, Any] | Any) -> str:
    if isinstance(outcome, Mapping):
        status = outcome.get("status", "Unknown")
        db_error = outcome.get("db_error")
    else:
        status = getattr(outcome, "status", "Unknown")
        db_error = getattr(outcome, "db_error", None)
    status_value = getattr(status, "value", status)
    return f"status={status_value}; db_error={db_error or '(none)'}"


def build_reflection_prompt(
    dataset_name: str,
    question: str,
    schema: str,
    attempts: Sequence[str],
    outcomes: Sequence[Mapping[str, Any] | Any],
    *,
    current_error_type: str,
    evidence: str = "",
) -> Prompt:
    """Define the Reflexion generation call after an unsuccessful attempt."""

    if not attempts or len(attempts) != len(outcomes):
        raise ValueError("reflection requires one public outcome per attempt")
    dataset = _validate_dataset(dataset_name)
    version = prompt_version(dataset, "reflection")
    context = _dataset_context(dataset, evidence)
    history = "\n\n".join(
        f"[Attempt {index}]\nSQL: {sql}\nOutcome: {_outcome_text(outcome)}"
        for index, (sql, outcome) in enumerate(zip(attempts, outcomes), start=1)
    )
    text = (
        f"PROMPT_VERSION: {version}\n"
        "Write a concise debugging reflection using only the observed attempt outcomes.\n"
        "Do not claim an unobserved cause and do not produce the next SQL query.\n\n"
        "DATABASE SCHEMA:\n"
        f"{schema.strip()}\n\n"
        f"{context}\n"
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        f"CURRENT ERROR TYPE: {current_error_type or 'Unknown'}\n\n"
        "ATTEMPTS AND OBSERVED OUTCOMES:\n"
        f"{history}\n\n"
        "Put the reflection between <reflection> and </reflection> tags.\n"
    )
    return Prompt(dataset, "reflection", version, text)


__all__ = [
    "PROMPT_FORMAT_VERSION",
    "PROMPT_VERSIONS",
    "Prompt",
    "StaticExample",
    "build_initial_prompt",
    "build_reflection_prompt",
    "build_repair_prompt",
    "prompt_hash",
    "prompt_version",
]
