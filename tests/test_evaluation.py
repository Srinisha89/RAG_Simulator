import csv
import json
from datetime import date

import pytest

from rag_simulator.evaluation import (
    compute_metrics,
    load_golden_set,
    promote_feedback_to_case,
    run_default_evaluation,
    run_evaluation,
    write_evaluation_results,
)
from rag_simulator.retrieval import chunk_documents
from rag_simulator.safety import PolicyAgent


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_golden_set_has_schema_and_required_case_coverage():
    cases = load_golden_set(ROOT / "data" / "golden_set.json")

    assert len(cases) >= 30
    assert all(case.case_id and case.question and case.expected_decision for case in cases)
    tags = {tag for case in cases for tag in case.tags}
    assert {
        "happy-path",
        "exception",
        "stale",
        "restricted",
        "pii",
        "injection",
        "vague",
        "refusal",
        "escalation",
    }.issubset(tags)


def test_metric_calculations_for_retrieval_grounding_and_escalation():
    metrics = compute_metrics(
        [
            {
                "expected_decision": "answer",
                "actual_decision": "answer",
                "expected_doc_ids": ["POL-001", "POL-002"],
                "retrieved_doc_ids": ["POL-001", "POL-003"],
                "grounded": True,
                "permission_leak": False,
            },
            {
                "expected_decision": "escalate",
                "actual_decision": "answer",
                "expected_doc_ids": [],
                "retrieved_doc_ids": [],
                "grounded": False,
                "permission_leak": False,
            },
        ],
        k=2,
    )

    assert metrics.decision_accuracy == pytest.approx(0.5)
    assert metrics.context_precision == pytest.approx(0.5)
    assert metrics.context_recall == pytest.approx(0.5)
    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.grounded_answer_rate == pytest.approx(0.5)
    assert metrics.escalation_correctness == pytest.approx(0.5)
    assert metrics.permission_leaks == 0


def test_context_precision_ignores_cases_that_expect_no_retrieval():
    metrics = compute_metrics(
        [
            {
                "expected_decision": "answer",
                "actual_decision": "answer",
                "expected_doc_ids": ["POL-001"],
                "retrieved_doc_ids": ["POL-001", "POL-002"],
                "grounded": True,
                "permission_leak": False,
            },
            {
                "expected_decision": "escalate",
                "actual_decision": "escalate",
                "expected_doc_ids": [],
                "retrieved_doc_ids": [],
                "grounded": False,
                "permission_leak": False,
            },
        ]
    )

    assert metrics.context_precision == pytest.approx(0.5)
    assert metrics.grounded_answer_rate == pytest.approx(1.0)


def test_evaluation_writes_dated_csv_and_summary(tmp_path, policy_documents):
    cases = [
        {
            "case_id": "GOLD-TEST-001",
            "question": "What is the fictional extension request window? Passport number X1234567",
            "expected_decision": "answer",
            "expected_doc_ids": ["POL-001"],
            "user_groups": ["employees"],
            "tags": ["happy-path"],
        }
    ]
    agent = PolicyAgent(
        chunk_documents(policy_documents),
        {},
        today=date(2026, 10, 19),
    )

    rows = run_evaluation(agent, cases)
    csv_path, report_path = write_evaluation_results(
        rows,
        tmp_path,
        run_date=date(2026, 10, 19),
    )

    assert csv_path.name == "evaluation_2026-10-19.csv"
    assert report_path.name == "summary_2026-10-19.json"
    with csv_path.open(newline="", encoding="utf-8") as result_file:
        saved_rows = list(csv.DictReader(result_file))
    assert len(saved_rows) == 1
    assert saved_rows[0]["actual_decision"] == "answer"
    assert "X1234567" not in saved_rows[0]["question"]
    assert saved_rows[0]["pii_detected"] == "True"
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["metrics"]["permission_leaks"] == 0
    assert "rollout_simulation" in summary
    assert summary["cost_scenario"]["target_usd_per_query"] == 0.01


def test_feedback_promotion_creates_redacted_golden_case(tmp_path):
    feedback = {
        "answer_id": "ANS-009",
        "rating": "down",
        "reason": "should-have-escalated",
        "question": "Passport X1234567: can I travel during a pending extension?",
    }

    case = promote_feedback_to_case(feedback, tmp_path / "golden_set.json")

    assert case["case_id"] == "GOLD-FEEDBACK-ANS-009"
    assert case["expected_decision"] == "escalate"
    assert case["question"].find("X1234567") == -1
    saved = json.loads((tmp_path / "golden_set.json").read_text(encoding="utf-8"))
    assert saved == [case]


def test_feedback_promotion_deduplicates_answer_id(tmp_path):
    path = tmp_path / "golden_set.json"
    original = promote_feedback_to_case(
        {
            "answer_id": "ANS-010",
            "rating": "down",
            "reason": "wrong",
            "question": "What is the fictional extension process?",
        },
        path,
    )
    repeated = promote_feedback_to_case(
        {
            "answer_id": "ANS-010",
            "rating": "down",
            "reason": "outdated",
            "question": "What is the fictional extension process?",
        },
        path,
    )

    assert repeated == original
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 1


def test_default_golden_evaluation_has_zero_permission_leaks_and_writes_report(tmp_path):
    rows, summary, csv_path, report_path = run_default_evaluation(
        project_root=ROOT,
        output_directory=tmp_path,
        run_date=date(2026, 10, 19),
    )

    assert len(rows) == 30
    assert summary["metrics"]["permission_leaks"] == 0
    assert csv_path.exists()
    assert report_path.exists()