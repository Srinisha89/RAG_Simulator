from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from rag_simulator.observability import require_admin
from rag_simulator.retrieval import chunk_documents
from rag_simulator.safety import PolicyAgent, detect_and_redact_pii


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    question: str
    expected_decision: str
    expected_doc_ids: list[str]
    user_groups: list[str]
    tags: list[str]


@dataclass(frozen=True)
class EvaluationMetrics:
    decision_accuracy: float
    context_precision: float
    context_recall: float
    recall_at_k: float
    grounded_answer_rate: float
    escalation_correctness: float
    permission_leaks: int


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    raw_cases = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    seen_ids = set()
    for raw in raw_cases:
        required = {
            "case_id",
            "question",
            "expected_decision",
            "expected_doc_ids",
            "user_groups",
            "tags",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Golden case is missing fields: {', '.join(sorted(missing))}")
        if raw["case_id"] in seen_ids:
            raise ValueError(f"Duplicate golden case id: {raw['case_id']}")
        if raw["expected_decision"] not in {"answer", "clarify", "escalate"}:
            raise ValueError(f"Invalid expected decision in {raw['case_id']}")
        seen_ids.add(raw["case_id"])
        cases.append(
            GoldenCase(
                case_id=raw["case_id"],
                question=raw["question"],
                expected_decision=raw["expected_decision"],
                expected_doc_ids=list(raw["expected_doc_ids"]),
                user_groups=list(raw["user_groups"]),
                tags=list(raw["tags"]),
            )
        )
    return cases


def compute_metrics(rows: Sequence[dict[str, Any]], *, k: int = 5) -> EvaluationMetrics:
    if not rows:
        raise ValueError("Cannot compute evaluation metrics for an empty result set.")
    decision_accuracy = sum(
        row["actual_decision"] == row["expected_decision"] for row in rows
    ) / len(rows)
    escalation_correctness = sum(
        (row["actual_decision"] == "escalate")
        == (row["expected_decision"] == "escalate")
        for row in rows
    ) / len(rows)
    context_precisions = []
    context_recalls = []
    recall_at_k_values = []
    for row in rows:
        expected = set(row["expected_doc_ids"])
        retrieved = list(row["retrieved_doc_ids"])
        hits = expected.intersection(retrieved)
        if expected and retrieved:
            context_precisions.append(len(hits) / len(retrieved))
        if expected:
            context_recalls.append(len(hits) / len(expected))
            top_hits = expected.intersection(retrieved[:k])
            recall_at_k_values.append(len(top_hits) / len(expected))
    answer_rows = [row for row in rows if row["actual_decision"] == "answer"]
    grounded_answer_rate = (
        sum(bool(row["grounded"]) for row in answer_rows) / len(answer_rows)
        if answer_rows
        else 0.0
    )
    return EvaluationMetrics(
        decision_accuracy=decision_accuracy,
        context_precision=sum(context_precisions) / len(context_precisions) if context_precisions else 0.0,
        context_recall=sum(context_recalls) / len(context_recalls) if context_recalls else 0.0,
        recall_at_k=sum(recall_at_k_values) / len(recall_at_k_values) if recall_at_k_values else 0.0,
        grounded_answer_rate=grounded_answer_rate,
        escalation_correctness=escalation_correctness,
        permission_leaks=sum(bool(row["permission_leak"]) for row in rows),
    )


def run_evaluation(
    agent: PolicyAgent,
    cases: Sequence[GoldenCase | dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for raw_case in cases:
        case = asdict(raw_case) if isinstance(raw_case, GoldenCase) else raw_case
        result = agent.respond(
            case["question"],
            user_groups=set(case["user_groups"]),
            user_id=f"EVAL-{case['case_id']}",
        )
        safe_question, pii_detected = detect_and_redact_pii(case["question"])
        retrieved_doc_ids = list(dict.fromkeys(chunk.document_id for chunk in result.retrieved))
        permitted_groups = set(case["user_groups"])
        permission_leak = any(
            not chunk.access_group_ids.intersection(permitted_groups)
            for chunk in result.retrieved
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "question": safe_question,
                "pii_detected": pii_detected,
                "tags": ",".join(case["tags"]),
                "expected_decision": case["expected_decision"],
                "actual_decision": result.status,
                "expected_doc_ids": case["expected_doc_ids"],
                "retrieved_doc_ids": retrieved_doc_ids,
                "citation": result.citation,
                "grounded": bool(result.status == "answer" and result.citation and not result.unsupported_claims),
                "permission_leak": permission_leak,
                "gate_decision": result.gate_decision,
                "unsupported_claims": result.unsupported_claims,
            }
        )
    return rows


def write_evaluation_results(
    rows: Sequence[dict[str, Any]],
    output_directory: str | Path,
    *,
    run_date: date | None = None,
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("Cannot write results for an empty evaluation.")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    current_day = run_date or date.today()
    date_label = current_day.isoformat()
    csv_path = output_path / f"evaluation_{date_label}.csv"
    report_path = output_path / f"summary_{date_label}.json"
    metrics = compute_metrics(rows)
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    report = {
        "run_date": date_label,
        **evaluation_summary(rows),
        "note": "Deterministic synthetic-data evaluation; not an attorney correctness review.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return csv_path, report_path


def promote_feedback_to_case(
    feedback: dict[str, Any],
    golden_set_path: str | Path,
    *,
    user_groups: Sequence[str] = ("employees",),
) -> dict[str, Any]:
    if feedback.get("rating") != "down":
        raise ValueError("Only negative feedback can be promoted to a golden case.")
    answer_id = str(feedback["answer_id"])
    path = Path(golden_set_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    case_id = f"GOLD-FEEDBACK-{answer_id}"
    existing = next((case for case in cases if case["case_id"] == case_id), None)
    if existing is not None:
        return existing
    question, _ = detect_and_redact_pii(str(feedback["question"]))
    reason = feedback.get("reason", "wrong")
    expected_decision = "escalate" if reason == "should-have-escalated" else "answer"
    case = {
        "case_id": case_id,
        "question": question,
        "expected_decision": expected_decision,
        "expected_doc_ids": [],
        "user_groups": list(user_groups),
        "tags": ["feedback", reason],
        "feedback_answer_id": answer_id,
    }
    cases.append(case)
    path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    return case


def evaluation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = asdict(compute_metrics(rows))
    faithfulness_target = 1.0
    alpha_pass = (
        metrics["grounded_answer_rate"] >= faithfulness_target
        and metrics["permission_leaks"] == 0
    )
    beta_pass = (
        alpha_pass
        and metrics["escalation_correctness"] == 1.0
    )
    ga_pass = (
        beta_pass
        and metrics["decision_accuracy"] >= 0.90
    )
    uncached_cost = (4000 * 3.0 + 500 * 15.0 + 250 * 0.25 + 40 * 1.25) / 1_000_000
    cached_cost = (4000 * 0.1 * 3.0 + 500 * 15.0 + 250 * 0.1 * 0.25 + 40 * 1.25) / 1_000_000
    return {
        "case_count": len(rows),
        "metrics": metrics,
        "cost_scenario": {
            "assumption": "Illustrative 90% discount on all input tokens; not measured against live API billing and likely optimistic.",
            "uncached_usd_per_query": uncached_cost,
            "cached_usd_per_query": cached_cost,
            "target_usd_per_query": 0.010,
        },
        "rollout_simulation": {
            "internal_alpha": {"pass": alpha_pass, "requirement": "Grounded-answer rate 100% and zero permission leaks."},
            "closed_beta": {"pass": beta_pass, "requirement": "Alpha gate plus 100% escalation correctness."},
            "general_availability": {"pass": ga_pass, "requirement": "Beta gate plus at least 90% decision accuracy."},
            "warning": "Simulated only. These synthetic fixtures do not establish attorney-reviewed correctness or deployment readiness.",
        },
    }


def read_results_for_admin(path: str | Path, *, role: str) -> list[dict[str, str]]:
    require_admin(role)
    with Path(path).open(newline="", encoding="utf-8") as result_file:
        return list(csv.DictReader(result_file))


def run_default_evaluation(
    *,
    project_root: str | Path,
    output_directory: str | Path | None = None,
    run_date: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, Path]:
    root = Path(project_root)
    documents = json.loads((root / "data" / "policy_documents.json").read_text(encoding="utf-8"))
    latest_versions = json.loads((root / "data" / "latest_versions.json").read_text(encoding="utf-8"))
    cases = load_golden_set(root / "data" / "golden_set.json")
    agent = PolicyAgent(
        chunk_documents(documents),
        latest_versions,
        today=run_date,
    )
    rows = run_evaluation(agent, cases)
    summary = evaluation_summary(rows)
    csv_path, report_path = write_evaluation_results(
        rows,
        output_directory or root / "eval_results",
        run_date=run_date,
    )
    return rows, summary, csv_path, report_path