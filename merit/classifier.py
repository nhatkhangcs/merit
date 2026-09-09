"""High-precision DBMS classification and conservative semantic hypotheses."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .feedback import Outcome, OutcomeStatus


@dataclass(frozen=True)
class ErrorClassification:
    """A typed observation with explicit evidence and provenance."""

    error_type: str
    error_subtype: str
    evidence: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


UNKNOWN = ErrorClassification(
    error_type="Unknown",
    error_subtype="Unknown",
    evidence="No high-precision rule matched.",
    source="none",
)


_DBMS_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bno such table:\s*(.+)", re.IGNORECASE),
        "Schema Linking",
        "Missing Table",
    ),
    (
        re.compile(r"\bno such column:\s*(.+)", re.IGNORECASE),
        "Schema Linking",
        "Missing Column",
    ),
    (
        re.compile(r"\bno such view:\s*(.+)", re.IGNORECASE),
        "Schema Linking",
        "Missing View",
    ),
    (
        re.compile(r"\bambiguous column name:\s*(.+)", re.IGNORECASE),
        "Schema Linking",
        "Ambiguous Column",
    ),
    (
        re.compile(r"\bno such function:\s*(.+)", re.IGNORECASE),
        "Syntax",
        "Unknown Function",
    ),
    (
        re.compile(
            r"(?:\bsyntax error\b|\bincomplete input\b|\bunrecognized token\b)",
            re.IGNORECASE,
        ),
        "Syntax",
        "Parse Failure",
    ),
    (
        re.compile(
            r"(?:misuse of aggregate|aggregate functions are not allowed)",
            re.IGNORECASE,
        ),
        "Aggregation",
        "DBMS Aggregate Misuse",
    ),
    (
        re.compile(r"(?:datatype mismatch|type mismatch)", re.IGNORECASE),
        "Filter/Value",
        "Type Mismatch",
    ),
    (
        re.compile(
            r"(?:attempt to write a readonly database|only SELECT or WITH queries)",
            re.IGNORECASE,
        ),
        "Execution",
        "Read Only Violation",
    ),
    (
        re.compile(r"\bdatabase not found\b", re.IGNORECASE),
        "Execution",
        "DB Not Found",
    ),
)


def classify_dbms_error(error_message: str | None) -> ErrorClassification:
    """Classify only directly observed, high-precision DBMS messages."""

    message = (error_message or "").strip()
    if not message:
        return UNKNOWN
    if message.upper() == "TIMEOUT" or "timed out" in message.lower():
        return ErrorClassification(
            "Execution",
            "Timeout",
            message,
            "dbms",
        )
    for pattern, error_type, subtype in _DBMS_RULES:
        if pattern.search(message):
            return ErrorClassification(
                error_type,
                subtype,
                message,
                "dbms",
            )
    return ErrorClassification(
        "Unknown",
        "Unknown",
        message,
        "dbms",
    )


def _outcome_field(outcome: Outcome | Mapping[str, Any], name: str) -> Any:
    if isinstance(outcome, Mapping):
        return outcome.get(name)
    return getattr(outcome, name)


def hypothesize_semantic_error(
    sql_ast: Any,
    outcome: Outcome | Mapping[str, Any],
) -> ErrorClassification:
    """Return only a bounded hypothesis for an executable unconfirmed mismatch.

    ``sql_ast`` is accepted for a stable classifier API, but SQL shape alone is
    intentionally insufficient evidence for a specific semantic error.
    """

    del sql_ast
    status = _outcome_field(outcome, "status")
    status_value = getattr(status, "value", status)
    predicted_exec_ok = bool(_outcome_field(outcome, "predicted_exec_ok"))
    if (
        predicted_exec_ok
        and status_value == OutcomeStatus.DENOTATION_MISMATCH.value
    ):
        return ErrorClassification(
            "Result Mismatch",
            "Unknown",
            "The query executed but was not confirmed correct; no reliable "
            "semantic cause was observed.",
            "semantic_hypothesis",
        )
    return UNKNOWN


def classify_outcome(
    sql_ast: Any,
    outcome: Outcome | Mapping[str, Any],
) -> ErrorClassification:
    """Classify observed DBMS failures before considering a semantic hypothesis."""

    db_error = _outcome_field(outcome, "db_error")
    if db_error:
        return classify_dbms_error(str(db_error))
    return hypothesize_semantic_error(sql_ast, outcome)


__all__ = [
    "ErrorClassification",
    "UNKNOWN",
    "classify_dbms_error",
    "classify_outcome",
    "hypothesize_semantic_error",
]
