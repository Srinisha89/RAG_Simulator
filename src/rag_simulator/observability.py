from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rag_simulator.safety import detect_and_redact_pii


def require_admin(role: str) -> None:
    if role != "admin":
        raise PermissionError("Only the admin role may view trace records.")


@dataclass(frozen=True)
class CostEstimate:
    main: float
    gate: float

    @property
    def total(self) -> float:
        return self.main + self.gate


class UsageLedger:
    def __init__(self):
        self._costs: dict[str, float] = {}

    def record(self, user_id: str, cost: CostEstimate) -> float:
        self._costs[user_id] = self._costs.get(user_id, 0.0) + cost.total
        return self._costs[user_id]

    def for_user(self, user_id: str) -> float:
        return self._costs.get(user_id, 0.0)

    @property
    def total(self) -> float:
        return sum(self._costs.values())


def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    gate_input_tokens: int,
    gate_output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
    gate_input_price_per_million: float,
    gate_output_price_per_million: float,
) -> CostEstimate:
    main = (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000
    gate = (
        gate_input_tokens * gate_input_price_per_million
        + gate_output_tokens * gate_output_price_per_million
    ) / 1_000_000
    return CostEstimate(main=main, gate=gate)


class TraceLogger:
    def __init__(self, path: str | Path, *, retention_days: int):
        if retention_days < 1:
            raise ValueError("Trace retention must be at least one day.")
        self.path = Path(path)
        self.retention_days = retention_days

    def log(self, record: dict[str, Any]) -> None:
        safe_record = dict(record)
        for field in ("question", "answer", "context"):
            value = safe_record.get(field)
            if isinstance(value, str):
                safe_record[field], _ = detect_and_redact_pii(value)
        safe_record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(safe_record, ensure_ascii=True) + "\n")

    def read(self, *, role: str, today: date | None = None) -> list[dict[str, Any]]:
        require_admin(role)
        if not self.path.exists():
            return []
        current_day = today or date.today()
        cutoff = current_day - timedelta(days=self.retention_days)
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            created = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
            if created.date() >= cutoff:
                records.append(record)
        return records

    def prune(self, *, today: date | None = None) -> int:
        if not self.path.exists():
            return 0
        current_day = today or date.today()
        cutoff = current_day - timedelta(days=self.retention_days)
        kept_lines = []
        removed = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            created = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
            if created.date() >= cutoff:
                kept_lines.append(json.dumps(record, ensure_ascii=True))
            else:
                removed += 1
        contents = "\n".join(kept_lines)
        self.path.write_text(contents + ("\n" if contents else ""), encoding="utf-8")
        return removed


class FeedbackStore:
    ALLOWED_REASONS = {"wrong", "outdated", "off-topic", "should-have-escalated"}

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, *, answer_id: str, rating: str, reason: str, question: str) -> dict[str, str]:
        if rating not in {"up", "down"}:
            raise ValueError("Feedback rating must be up or down.")
        if rating == "down" and reason not in self.ALLOWED_REASONS:
            raise ValueError("A diagnostic reason is required for negative feedback.")
        safe_question, _ = detect_and_redact_pii(question)
        record = {
            "answer_id": answer_id,
            "rating": rating,
            "reason": reason,
            "question": safe_question,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as feedback_file:
            feedback_file.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record