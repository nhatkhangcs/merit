"""Public feedback types shared by runners, prompts, and memory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .config import FEEDBACK_REGIMES


class FeedbackRegime(str, Enum):
    DENOTATION_CONFIRMED = "denotation_confirmed"
    DBMS_ONLY = "dbms_only"


class OutcomeStatus(str, Enum):
    CORRECT = "CORRECT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    DENOTATION_MISMATCH = "DENOTATION_MISMATCH"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class Outcome:
    """The exact public outcome contract for one attempted SQL query."""

    predicted_exec_ok: bool
    predicted_rows: list[Any] | None
    db_error: str | None
    oracle_correct: bool | None
    status: OutcomeStatus
    feedback_regime: str

    def __post_init__(self) -> None:
        if type(self.predicted_exec_ok) is not bool:
            raise TypeError("predicted_exec_ok must be bool")
        if self.predicted_rows is not None and not isinstance(self.predicted_rows, list):
            raise TypeError("predicted_rows must be a list or None")
        if self.db_error is not None and not isinstance(self.db_error, str):
            raise TypeError("db_error must be str or None")
        if self.oracle_correct is not None and type(self.oracle_correct) is not bool:
            raise TypeError("oracle_correct must be bool or None")
        status = (
            self.status
            if isinstance(self.status, OutcomeStatus)
            else OutcomeStatus(str(self.status))
        )
        regime = (
            self.feedback_regime.value
            if isinstance(self.feedback_regime, FeedbackRegime)
            else str(self.feedback_regime)
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "feedback_regime", regime)

        if regime not in FEEDBACK_REGIMES:
            raise ValueError(f"unsupported feedback regime: {regime}")
        if self.predicted_exec_ok and self.predicted_rows is None:
            raise ValueError("successful execution requires predicted_rows")
        if not self.predicted_exec_ok and self.predicted_rows is not None:
            raise ValueError("failed execution cannot expose predicted_rows")
        if status == OutcomeStatus.CORRECT:
            if not self.predicted_exec_ok or self.oracle_correct is not True:
                raise ValueError("CORRECT requires successful, oracle-confirmed execution")
            if regime != FeedbackRegime.DENOTATION_CONFIRMED.value:
                raise ValueError("dbms_only outcomes can never be CORRECT")
            if self.db_error is not None:
                raise ValueError("CORRECT cannot carry db_error")
        if status == OutcomeStatus.DENOTATION_MISMATCH:
            if not self.predicted_exec_ok:
                raise ValueError("DENOTATION_MISMATCH requires executable SQL")
            if self.db_error is not None:
                raise ValueError("DENOTATION_MISMATCH cannot carry db_error")
            expected_oracle = (
                False
                if regime == FeedbackRegime.DENOTATION_CONFIRMED.value
                else None
            )
            if self.oracle_correct is not expected_oracle:
                raise ValueError(
                    "DENOTATION_MISMATCH oracle flag conflicts with feedback regime"
                )
        if status in {OutcomeStatus.EXECUTION_ERROR, OutcomeStatus.TIMEOUT}:
            if self.predicted_exec_ok:
                raise ValueError(f"{status.value} cannot have successful execution")
            if not self.db_error:
                raise ValueError(f"{status.value} requires db_error")
            expected_oracle = (
                False
                if regime == FeedbackRegime.DENOTATION_CONFIRMED.value
                else None
            )
            if self.oracle_correct is not expected_oracle:
                raise ValueError(
                    f"{status.value} oracle flag conflicts with feedback regime"
                )
        if regime == FeedbackRegime.DBMS_ONLY.value and self.oracle_correct is not None:
            raise ValueError("dbms_only must not expose oracle correctness")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping with exactly the six Outcome fields."""

        return {
            "predicted_exec_ok": self.predicted_exec_ok,
            "predicted_rows": self.predicted_rows,
            "db_error": self.db_error,
            "oracle_correct": self.oracle_correct,
            "status": self.status.value,
            "feedback_regime": self.feedback_regime,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Outcome":
        required = {
            "predicted_exec_ok",
            "predicted_rows",
            "db_error",
            "oracle_correct",
            "status",
            "feedback_regime",
        }
        missing = required.difference(value)
        extra = set(value).difference(required)
        if missing or extra:
            raise ValueError(
                f"Outcome fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if type(value["predicted_exec_ok"]) is not bool:
            raise TypeError("predicted_exec_ok must be bool")
        return cls(
            predicted_exec_ok=value["predicted_exec_ok"],
            predicted_rows=value["predicted_rows"],
            db_error=value["db_error"],
            oracle_correct=value["oracle_correct"],
            status=OutcomeStatus(str(value["status"])),
            feedback_regime=str(value["feedback_regime"]),
        )


__all__ = [
    "FeedbackRegime",
    "Outcome",
    "OutcomeStatus",
]
