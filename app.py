import json
import os
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
from rag_simulator.retrieval import (
    check_freshness,
    chunk_documents,
    format_citation,
    LocalPolicyIndex,
    search_policy,
)
from rag_simulator.safety import (
    EscalationQueue,
    PolicyAgent,
    check_deadline,
    detect_and_redact_pii,
    get_my_profile,
)
from rag_simulator.observability import (
    FeedbackStore,
    TraceLogger,
    UsageLedger,
    estimate_cost,
    require_admin,
)
from rag_simulator.evaluation import (
    promote_feedback_to_case,
    run_default_evaluation,
)


ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

st.set_page_config(page_title="Policy Assistant | Fictional Demo", page_icon="F", layout="centered")
st.title("Immigration Policy Assistant")
st.warning(
    "FICTIONAL DATA | NOT LEGAL ADVICE\n\n"
    "Portfolio simulation only. All profiles and policy documents are synthetic. "
    "This demo does not provide current USCIS or legal guidance."
)

role = st.sidebar.selectbox("Demo role", ["employee", "immigration_team", "admin"])
st.sidebar.caption("Role switching is a portfolio simulation, not authentication.")

employees = json.loads((ROOT / "data" / "employees.json").read_text(encoding="utf-8"))
documents = json.loads((ROOT / "data" / "policy_documents.json").read_text(encoding="utf-8"))
latest_versions = json.loads((ROOT / "data" / "latest_versions.json").read_text(encoding="utf-8"))
chunks = chunk_documents(documents)
embedding_model = os.environ.get("POLICY_EMBEDDING_MODEL")
chroma_path = os.environ.get("POLICY_CHROMA_PATH", str(ROOT / "chroma_data"))


@st.cache_resource
def load_dense_index(model_path: str, persistence_path: str):
    return LocalPolicyIndex(persistence_path, model_path)


try:
    dense_index = load_dense_index(embedding_model, chroma_path) if embedding_model else None
    if dense_index is not None and dense_index._collection.count() == 0:
        dense_index.refresh(chunks)
    dense_index_error = None
except (RuntimeError, OSError, ValueError) as error:
    dense_index = None
    dense_index_error = str(error)
st.sidebar.metric("Synthetic profiles", len(employees))
st.sidebar.metric("Fictional policy documents", len(documents))
st.sidebar.caption("Dense retrieval: local model active" if dense_index else "Dense retrieval: not configured")
st.sidebar.toggle(
    "Anthropic live mode (disabled)",
    value=False,
    disabled=True,
    help="Disabled to prevent Anthropic API charges.",
)
answerer = None
live_model_enabled = False
model_configuration_error = None

trace_logger = TraceLogger(
    ROOT / os.getenv("TRACE_PATH", "runtime/traces.jsonl"),
    retention_days=int(os.getenv("TRACE_RETENTION_DAYS", "30")),
)
feedback_store = FeedbackStore(ROOT / os.getenv("FEEDBACK_PATH", "runtime/feedback.jsonl"))

st.subheader("Ask a policy question")
st.info(
    "Safety and evaluation controls are active. Answers describe fictional company policy only. "
    "Questions about real immigration law or personal outcomes are routed for human review."
)
if role == "employee":
    selected_employee = get_my_profile(
        "SYN-001", employees, requester_id="SYN-001", role=role
    )
    st.sidebar.caption(f"Signed in as {selected_employee['name']} (synthetic demo profile)")
    user_groups = set(selected_employee["group_ids"])
else:
    selected_employee = st.selectbox(
        "Synthetic profile",
        employees,
        format_func=lambda employee: f"{employee['employee_id']} | {employee['name']}",
    )
    user_groups = (
        {"employees", "immigration-team"}
        if role == "immigration_team"
        else {group for document in documents for group in document["access_group_ids"]}
    )

if role == "immigration_team":
    user_groups = {"employees", "immigration-team"}

def redact_question_input() -> None:
    safe_text, detected = detect_and_redact_pii(st.session_state.question_input)
    st.session_state.question_input = safe_text
    st.session_state.input_pii_detected = detected


if "phase3_queue" not in st.session_state:
    st.session_state.phase3_queue = EscalationQueue()
if "phase3_result" not in st.session_state:
    st.session_state.phase3_result = None
if "input_pii_detected" not in st.session_state:
    st.session_state.input_pii_detected = False
if "usage_ledger" not in st.session_state:
    st.session_state.usage_ledger = UsageLedger()
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()
trace_logger.prune()

agent = PolicyAgent(
    chunks,
    latest_versions,
    retriever=lambda query, available_chunks, *, user_groups, source_precedence, **kwargs: search_policy(
        query,
        available_chunks,
        user_groups=user_groups,
        source_precedence=source_precedence,
        dense_search=dense_index.search if dense_index else None,
    ),
    queue=st.session_state.phase3_queue,
    answerer=answerer,
)

question = st.text_input(
    "Question",
    key="question_input",
    on_change=redact_question_input,
    placeholder="Try: What is the fictional extension workflow?",
)
if st.session_state.input_pii_detected:
    st.warning("Sensitive-looking text was redacted. Please do not enter passport numbers or case identifiers.")

if st.button("Ask assistant", disabled=not question.strip()):
    if live_model_enabled and model_configuration_error:
        st.error("Live mode is enabled but Anthropic configuration is incomplete. No model call was made.")
        st.stop()
    if dense_index_error:
        st.warning(f"Local dense index unavailable: {dense_index_error}")
    result = agent.respond(
        question,
        user_groups=user_groups,
        user_id=selected_employee["employee_id"],
    )
    st.session_state.phase3_result = result
    usage = result.model_usage
    query_cost = estimate_cost(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        gate_input_tokens=int(usage.get("gate_input_tokens", 0)),
        gate_output_tokens=int(usage.get("gate_output_tokens", 0)),
        input_price_per_million=float(os.getenv("MAIN_INPUT_PRICE_PER_MILLION", "3.0")),
        output_price_per_million=float(os.getenv("MAIN_OUTPUT_PRICE_PER_MILLION", "15.0")),
        gate_input_price_per_million=float(os.getenv("FAST_INPUT_PRICE_PER_MILLION", "0.25")),
        gate_output_price_per_million=float(os.getenv("FAST_OUTPUT_PRICE_PER_MILLION", "1.25")),
    )
    user_cost = st.session_state.usage_ledger.record(selected_employee["employee_id"], query_cost)
    trace_logger.log(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question": result.redacted_question,
            "retrieved": [
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "version": chunk.version,
                    "section": chunk.section,
                    "passage": chunk.text,
                    "rank": rank,
                }
                for rank, chunk in enumerate(result.retrieved, start=1)
            ],
            "context_sent": [chunk.text for chunk in result.retrieved] if answerer else [],
            "answer": result.answer,
            "gate_decision": result.status,
            "gate_confidence": result.gate_decision,
            "unsupported_claims": result.unsupported_claims,
            "model_usage": usage,
            "query_cost_estimate": query_cost.total,
            "user_cost_estimate": user_cost,
            "role": role,
        }
    )

result = st.session_state.phase3_result
if result is not None:
    if result.status == "answer":
        st.markdown("#### Fictional policy information")
        st.write(result.message)
        st.caption(f"Source: {result.citation}")
        feedback_key = f"feedback_{id(result)}"
        rating = st.feedback("thumbs", key=feedback_key)
        if rating is not None and feedback_key not in st.session_state.feedback_submitted:
            reason = ""
            if rating == 0:
                reason = st.selectbox(
                    "What should be improved?",
                    ["wrong", "outdated", "off-topic", "should-have-escalated"],
                )
                promote_case = st.checkbox(
                    "Confirm this issue and add it to the golden evaluation set",
                    key=f"promote_{feedback_key}",
                )
                if not st.button("Submit feedback"):
                    reason = ""
            if rating == 1 or reason:
                feedback_record = feedback_store.record(
                    answer_id=f"ANS-{id(result)}",
                    rating="up" if rating == 1 else "down",
                    reason=reason,
                    question=result.redacted_question,
                )
                if rating == 0 and promote_case:
                    case = promote_feedback_to_case(
                        feedback_record,
                        ROOT / "data" / "golden_set.json",
                        user_groups=sorted(user_groups),
                    )
                    st.info(f"Confirmed issue added as {case['case_id']}.")
                st.session_state.feedback_submitted.add(feedback_key)
                st.success("Feedback recorded with sensitive-looking text redacted.")
    elif result.status == "clarify":
        st.info(result.message)
    else:
        st.warning(result.message)
        if result.escalation_draft:
            with st.expander("Review escalation email draft"):
                st.write(f"To: {result.escalation_draft['to']}")
                st.write(f"Reason: {result.escalation_draft['reason']}")
                st.write(f"Question: {result.escalation_draft['question']}")
                if result.escalation_draft["findings"]:
                    st.write("Findings:")
                    for finding in result.escalation_draft["findings"]:
                        st.write(f"- {finding}")
            if result.escalation_id is None and st.button("Confirm and add to team queue"):
                st.session_state.phase3_result = agent.confirm_escalation(
                    result,
                    user_id=selected_employee["employee_id"],
                )
                st.rerun()

if st.session_state.phase3_queue.items and role == "immigration_team":
    st.subheader("Confirmed escalation queue")
    for item in st.session_state.phase3_queue.items:
        with st.container(border=True):
            st.markdown(f"**{item.escalation_id} | {item.user_id} | {item.created_on.isoformat()}**")
            st.write(item.reason)
            st.write(item.question)
            for finding in item.findings:
                st.caption(finding)

with st.expander("Fictional deadline calculator"):
    deadline = check_deadline(selected_employee, days_before_expiry=120)
    st.write(f"Profile expiry date: {deadline.profile_date.isoformat()}")
    st.write(f"Fictional request target date: {deadline.target_date.isoformat()}")
    st.caption(f"Rule: {deadline.rule} This is a synthetic demonstration, not a real filing deadline.")

if role == "admin":
    try:
        require_admin(role)
        trace_records = trace_logger.read(role=role)
    except (PermissionError, ValueError):
        trace_records = []
    with st.expander("Admin query trace"):
        if trace_records:
            st.json(trace_records[-1])
        else:
            st.write("No trace records have been recorded yet.")
    with st.expander("Token cost estimate"):
        st.metric("Estimated cost for this session", f"${st.session_state.usage_ledger.total:.4f}")
        st.caption("Per-user estimates are kept in this demo session and derived from reported token usage.")
    with st.expander("Golden-set evaluation"):
        st.caption("This deterministic run uses only synthetic fixtures and does not make Anthropic API calls.")
        if st.button("Run 30-case evaluation"):
            rows, summary, csv_path, report_path = run_default_evaluation(project_root=ROOT)
            st.session_state.last_evaluation = {
                "rows": rows,
                "summary": summary,
                "csv_path": str(csv_path.relative_to(ROOT)),
                "report_path": str(report_path.relative_to(ROOT)),
            }
        evaluation = st.session_state.get("last_evaluation")
        if evaluation:
            st.json(evaluation["summary"])
            st.caption(f"CSV: {evaluation['csv_path']} | Summary: {evaluation['report_path']}")
            failures = [
                row for row in evaluation["rows"]
                if row["expected_decision"] != row["actual_decision"]
                or (row["expected_doc_ids"] and not set(row["expected_doc_ids"]).intersection(row["retrieved_doc_ids"]))
                or row["permission_leak"]
            ]
            if failures:
                st.markdown("**Cases to review**")
                for failure in failures:
                    st.write(
                        f"{failure['case_id']}: expected {failure['expected_decision']}, "
                        f"got {failure['actual_decision']}; retrieved {failure['retrieved_doc_ids']}"
                    )
            st.download_button(
                "Download evaluation CSV",
                Path(evaluation["csv_path"]).read_bytes(),
                file_name=Path(evaluation["csv_path"]).name,
                mime="text/csv",
            )

st.sidebar.metric("Session model cost", f"${st.session_state.usage_ledger.total:.4f}")
st.sidebar.metric(
    "Current synthetic user cost",
    f"${st.session_state.usage_ledger.for_user(selected_employee['employee_id']):.4f}",
)

st.caption(f"Current demo role: {role}")