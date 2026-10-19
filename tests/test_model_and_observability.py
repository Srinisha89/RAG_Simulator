from datetime import date

import pytest

from rag_simulator.model import AnthropicAnswerer, ModelConfig, ModelConfigurationError
from rag_simulator.observability import (
    FeedbackStore,
    TraceLogger,
    UsageLedger,
    estimate_cost,
    require_admin,
)
from rag_simulator.safety import PolicyAgent
from rag_simulator.retrieval import chunk_documents


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def test_missing_anthropic_configuration_fails_closed():
    with pytest.raises(ModelConfigurationError):
        ModelConfig.from_values(api_key="", main_model="", fast_model="")


def test_answerer_uses_main_model_and_only_supplied_permission_filtered_evidence():
    response = type(
        "Response",
        (),
        {
            "content": [type("TextBlock", (), {"type": "text", "text": "The fictional process starts 120 days before expiry."})()],
            "usage": type("Usage", (), {"input_tokens": 150, "output_tokens": 20})(),
        },
    )()
    client = FakeClient(response)
    answerer = AnthropicAnswerer(
        ModelConfig(api_key="test-key", main_model="main-test", fast_model="fast-test"),
        client=client,
    )

    result = answerer.answer(
        "What is the extension window?",
        ["FICTIONAL source: start the request 120 days before expiry."],
    )

    call = client.messages.calls[0]
    assert call["model"] == "main-test"
    assert "FICTIONAL source" in call["messages"][0]["content"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert result.model == "main-test"
    assert result.input_tokens == 150


def test_fast_model_is_selected_only_for_gate_tasks():
    response = type(
        "Response",
        (),
        {
            "content": [type("TextBlock", (), {"type": "text", "text": "PASS"})()],
            "usage": type("Usage", (), {"input_tokens": 10, "output_tokens": 1})(),
        },
    )()
    client = FakeClient(response)
    answerer = AnthropicAnswerer(
        ModelConfig(api_key="test-key", main_model="main-test", fast_model="fast-test"),
        client=client,
    )

    answerer.classify("classify this")

    assert client.messages.calls[0]["model"] == "fast-test"
    assert client.messages.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_real_law_refusal_happens_before_model_call(policy_documents):
    client = FakeClient(None)
    answerer = AnthropicAnswerer(
        ModelConfig(api_key="test-key", main_model="main-test", fast_model="fast-test"),
        client=client,
    )
    agent = PolicyAgent(
        chunk_documents(policy_documents),
        {},
        answerer=answerer,
        today=date(2026, 10, 19),
    )

    result = agent.respond(
        "What is the current USCIS rule for my H-1B?",
        user_groups={"employees"},
    )

    assert result.status == "escalate"
    assert client.messages.calls == []


def test_model_answer_uses_grounding_judge_and_exposes_usage(policy_documents):
    answer_response = type(
        "Response",
        (),
        {
            "content": [type("TextBlock", (), {"type": "text", "text": "Employees start an extension request 120 days before the profile expiry date."})()],
            "usage": type("Usage", (), {"input_tokens": 100, "output_tokens": 18})(),
        },
    )()
    judge_response = type(
        "Response",
        (),
        {
            "content": [type("TextBlock", (), {"type": "text", "text": "PASS"})()],
            "usage": type("Usage", (), {"input_tokens": 50, "output_tokens": 1})(),
        },
    )()

    class SequenceClient:
        def __init__(self):
            self.messages = FakeMessages(answer_response)
            self.responses = [answer_response, judge_response]

            def create(**kwargs):
                self.messages.calls.append(kwargs)
                return self.responses.pop(0)

            self.messages.create = create

    client = SequenceClient()
    answerer = AnthropicAnswerer(
        ModelConfig(api_key="test-key", main_model="main-test", fast_model="fast-test"),
        client=client,
    )
    agent = PolicyAgent(
        chunk_documents(policy_documents),
        {},
        answerer=answerer,
        today=date(2026, 10, 19),
    )

    result = agent.respond(
        "What is the fictional extension request window?",
        user_groups={"employees"},
    )

    assert result.status == "answer"
    assert result.model_usage["input_tokens"] == 100
    assert result.gate_decision == "PASS"
    assert [call["model"] for call in client.messages.calls] == ["main-test", "fast-test"]


def test_unsupported_model_claim_escalates_even_if_fast_judge_passes(policy_documents):
    answer_response = type(
        "Response",
        (),
        {
            "content": [type("TextBlock", (), {"type": "text", "text": "Employees start an extension request 120 days before the profile expiry date. Approval is guaranteed."})()],
            "usage": type("Usage", (), {"input_tokens": 100, "output_tokens": 24})(),
        },
    )()
    judge_response = type(
        "Response",
        (),
        {
            "content": [type("TextBlock", (), {"type": "text", "text": "PASS"})()],
            "usage": type("Usage", (), {"input_tokens": 50, "output_tokens": 1})(),
        },
    )()

    class SequenceClient:
        def __init__(self):
            self.messages = FakeMessages(answer_response)
            self.responses = [answer_response, judge_response]

            def create(**kwargs):
                self.messages.calls.append(kwargs)
                return self.responses.pop(0)

            self.messages.create = create

    answerer = AnthropicAnswerer(
        ModelConfig(api_key="test-key", main_model="main-test", fast_model="fast-test"),
        client=SequenceClient(),
    )
    agent = PolicyAgent(
        chunk_documents(policy_documents), {}, answerer=answerer, today=date(2026, 10, 19)
    )

    result = agent.respond(
        "What is the fictional extension request window?",
        user_groups={"employees"},
    )

    assert result.status == "escalate"
    assert "Approval is guaranteed." in result.unsupported_claims
    assert not result.answer


def test_trace_redacts_pii_and_prunes_by_retention(tmp_path):
    logger = TraceLogger(tmp_path / "trace.jsonl", retention_days=30)
    logger.log(
        {
            "question": "Passport X1234567; receipt IOE1234567890",
            "answer": "synthetic only",
            "created_at": "2026-10-19T12:00:00+00:00",
        }
    )
    old_logger = TraceLogger(tmp_path / "old.jsonl", retention_days=30)
    old_logger.log(
        {"question": "old synthetic text", "created_at": "2026-01-01T00:00:00+00:00"}
    )

    records = logger.read(role="admin", today=date(2026, 10, 19))
    pruned = old_logger.prune(today=date(2026, 10, 19))

    assert len(records) == 1
    assert "X1234567" not in str(records[0])
    assert "IOE1234567890" not in str(records[0])
    assert pruned == 1
    assert old_logger.read(role="admin", today=date(2026, 10, 19)) == []


def test_trace_access_is_admin_only(tmp_path):
    logger = TraceLogger(tmp_path / "trace.jsonl", retention_days=30)

    with pytest.raises(PermissionError):
        logger.read(role="employee")

    with pytest.raises(PermissionError):
        require_admin("immigration_team")


def test_feedback_records_diagnostic_without_sensitive_question(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    record = store.record(
        answer_id="ANS-001",
        rating="down",
        reason="outdated",
        question="Passport X1234567: is the workflow current?",
    )

    assert record["reason"] == "outdated"
    assert "X1234567" not in record["question"]


def test_usage_cost_estimate_calculates_per_query_and_user():
    costs = estimate_cost(
        input_tokens=4000,
        output_tokens=500,
        gate_input_tokens=250,
        gate_output_tokens=40,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        gate_input_price_per_million=0.25,
        gate_output_price_per_million=1.25,
    )

    assert costs.total > 0
    assert costs.total == pytest.approx(costs.main + costs.gate)
    assert costs.main == pytest.approx(0.0195)


def test_usage_ledger_accumulates_cost_per_user():
    ledger = UsageLedger()
    query_cost = estimate_cost(
        input_tokens=1000,
        output_tokens=100,
        gate_input_tokens=0,
        gate_output_tokens=0,
        input_price_per_million=1.0,
        output_price_per_million=1.0,
        gate_input_price_per_million=0.0,
        gate_output_price_per_million=0.0,
    )

    ledger.record("SYN-001", query_cost)
    ledger.record("SYN-001", query_cost)
    ledger.record("SYN-002", query_cost)

    assert ledger.for_user("SYN-001") == pytest.approx(0.0022)
    assert ledger.total == pytest.approx(0.0033)