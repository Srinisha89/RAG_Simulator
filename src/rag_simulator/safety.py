from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Sequence

from rag_simulator.retrieval import (
    Freshness,
    PolicyChunk,
    check_freshness,
    format_citation,
    search_policy,
)


PII_PATTERNS = (
    re.compile(r"\bpassport(?:\s+number)?\s*[:#]?\s*(?:is\s+)?[A-Z0-9]{6,12}\b", re.I),
    re.compile(r"\b(?:receipt|case)(?:\s+number)?\s*[:#]?\s*(?:is\s+)?[A-Z]{3}\d{7,12}\b", re.I),
    re.compile(r"\b\d{9}\b"),
    re.compile(r"\b[Aa]-?\d{8,9}\b"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
)
INJECTION_PATTERNS = (
    re.compile(r"ignore (?:all )?(?:previous )?(?:rules|instructions)", re.I),
    re.compile(r"(?:reveal|show) (?:the )?(?:system prompt|hidden instructions)", re.I),
    re.compile(r"act as (?:an? )?(?:unrestricted|different) assistant", re.I),
)
REAL_LAW_TERMS = ("current uscis", "real uscis", "current law", "actual law", "recent policy changes")
OUTCOME_TERMS = ("will my", "will i be", "will they", "predict", "approved?", "approval odds")
PERSONAL_ADVICE_TERMS = ("should i", "recommend a visa", "which visa should", "what visa path")
HIGH_STAKES_TERMS = (
    "travel",
    "deadline",
    "expiry",
    "extension",
    "re-enter",
    "status",
    "spouse",
    "dependent",
    "work",
    "approval",
    "change employer",
    "change company",
)
INTERNAL_PROCESS_TERMS = (
    "company",
    "internal",
    "workflow",
    "process",
    "form",
    "request window",
    "department",
    "engineering",
    "approval",
    "legal advice",
    "model scope",
    "help desk",
    "checklist",
    "travel document",
)


@dataclass(frozen=True)
class GroundingResult:
    passed: bool
    unsupported_claims: list[str]


@dataclass(frozen=True)
class DeadlineResult:
    target_date: date
    profile_date: date
    days_before_expiry: int
    rule: str


@dataclass
class EscalationItem:
    escalation_id: str
    user_id: str
    reason: str
    question: str
    findings: list[str]
    created_on: date


@dataclass
class EscalationQueue:
    items: list[EscalationItem] = field(default_factory=list)

    def add(self, item: EscalationItem) -> None:
        self.items.append(item)


@dataclass
class AgentResult:
    status: str
    message: str
    answer: str = ""
    citation: str = ""
    retrieved: list[PolicyChunk] = field(default_factory=list)
    freshness: list[Freshness] = field(default_factory=list)
    pii_detected: bool = False
    redacted_question: str = ""
    injection_detected: bool = False
    escalation_draft: dict | None = None
    escalation_id: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    model_usage: dict[str, int | str] = field(default_factory=dict)
    gate_decision: str = "NOT_RUN"


def detect_and_redact_pii(text: str) -> tuple[str, bool]:
    redacted = text
    detected = False
    for pattern in PII_PATTERNS:
        redacted, count = pattern.subn("[REDACTED]", redacted)
        detected = detected or count > 0
    return redacted, detected


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def grounding_check(answer: str, passages: Sequence[str]) -> GroundingResult:
    evidence = " ".join(passages).lower()
    unsupported = []
    for claim in re.split(r"(?<=[.!?])\s+", answer.strip()):
        claim = claim.strip()
        if not claim:
            continue
        normalized = re.sub(r"[^a-z0-9 ]", " ", claim.lower())
        terms = [term for term in normalized.split() if len(term) > 3]
        if terms and not all(term in evidence for term in terms):
            unsupported.append(claim)
    return GroundingResult(not unsupported, unsupported)


def check_deadline(profile: dict, *, days_before_expiry: int) -> DeadlineResult:
    expiry = date.fromisoformat(profile["visa_expiry_date"])
    target = expiry - timedelta(days=days_before_expiry)
    return DeadlineResult(
        target_date=target,
        profile_date=expiry,
        days_before_expiry=days_before_expiry,
        rule=(
            f"Fictional company workflow: start the extension request "
            f"{days_before_expiry} days before the profile expiry date."
        ),
    )


def get_my_profile(
    employee_id: str,
    profiles: Sequence[dict],
    *,
    requester_id: str,
    role: str,
) -> dict:
    if role not in {"employee", "immigration_team", "admin"}:
        raise PermissionError("Unknown demo role.")
    if role == "employee" and employee_id != requester_id:
        raise PermissionError("Employees may view only their own synthetic profile.")
    profile = next(
        (entry for entry in profiles if entry["employee_id"] == employee_id),
        None,
    )
    if profile is None:
        raise KeyError(f"Synthetic profile {employee_id} was not found.")
    return dict(profile)


def explain_form_field(
    form_name: str,
    field_name: str,
    chunks: Sequence[PolicyChunk],
    user_groups: set[str] | frozenset[str],
) -> list[PolicyChunk]:
    return search_policy(
        f"{form_name} {field_name}",
        chunks,
        user_groups=user_groups,
        source_precedence="company_policy",
    )


class PolicyAgent:
    def __init__(
        self,
        chunks: Sequence[PolicyChunk],
        latest_versions: dict[str, str],
        *,
        today: date | None = None,
        retriever: Callable = search_policy,
        queue: EscalationQueue | None = None,
        answerer=None,
    ):
        self.chunks = list(chunks)
        self.latest_versions = latest_versions
        self.today = today or date.today()
        self.retriever = retriever
        self.queue = queue or EscalationQueue()
        self.answerer = answerer

    def confirm_escalation(self, draft: AgentResult, *, user_id: str) -> AgentResult:
        if draft.status != "escalate" or draft.escalation_draft is None:
            raise ValueError("Only an escalation draft can be confirmed.")
        escalation_id = f"ESC-{len(self.queue.items) + 1:04d}"
        details = draft.escalation_draft
        self.queue.add(
            EscalationItem(
                escalation_id=escalation_id,
                user_id=user_id,
                reason=details["reason"],
                question=details["question"],
                findings=list(details["findings"]),
                created_on=self.today,
            )
        )
        draft.escalation_id = escalation_id
        draft.message = f"Your question was added to the immigration team queue ({escalation_id})."
        return draft

    def respond(
        self,
        question: str,
        *,
        user_groups: set[str] | frozenset[str],
        user_id: str = "SYN-USER",
        confirm_escalation: bool = False,
    ) -> AgentResult:
        redacted_question, pii_detected = detect_and_redact_pii(question)
        injection_detected = detect_prompt_injection(redacted_question)
        lowered = redacted_question.lower()

        if injection_detected:
            return self._escalate(
                "The question contains instructions that cannot change the demo's safety rules.",
                redacted_question,
                user_id,
                user_groups,
                pii_detected,
                injection_detected,
                confirm_escalation,
            )

        tokens = re.findall(r"[a-z0-9-]+", lowered)
        vague_terms = {"visa", "immigration", "policy", "help", "question", "status"}
        if len(tokens) <= 1 or (len(tokens) <= 2 and set(tokens).issubset(vague_terms)):
            return AgentResult(
                status="clarify",
                message="What would you like to know about the fictional company policy?",
                pii_detected=pii_detected,
                redacted_question=redacted_question,
            )

        if any(term in lowered for term in REAL_LAW_TERMS):
            return self._escalate(
                "This demo contains fictional policy only and cannot answer current immigration law.",
                redacted_question,
                user_id,
                user_groups,
                pii_detected,
                injection_detected,
                confirm_escalation,
            )
        if any(term in lowered for term in OUTCOME_TERMS + PERSONAL_ADVICE_TERMS):
            return self._escalate(
                "The demo cannot predict case outcomes or recommend a personal visa path.",
                redacted_question,
                user_id,
                user_groups,
                pii_detected,
                injection_detected,
                confirm_escalation,
            )

        source_precedence = (
            "company_policy"
            if any(term in lowered for term in INTERNAL_PROCESS_TERMS)
            else "uscis_style_guidance"
        )
        retrieved = self.retriever(
            redacted_question,
            self.chunks,
            user_groups=user_groups,
            source_precedence=source_precedence,
        )
        if not retrieved:
            return self._escalate(
                "I could not find a clearly relevant permitted fictional source.",
                redacted_question,
                user_id,
                user_groups,
                pii_detected,
                injection_detected,
                confirm_escalation,
            )

        freshness = [
            check_freshness(chunk, self.latest_versions, today=self.today)
            for chunk in retrieved
        ]
        high_stakes = any(term in lowered for term in HIGH_STAKES_TERMS)
        if high_stakes and (freshness[0].is_stale or freshness[0].behind_latest):
            return self._escalate(
                "The relevant fictional source is stale or behind the listed latest version, so this high-impact question needs human review.",
                redacted_question,
                user_id,
                user_groups,
                pii_detected,
                injection_detected,
                confirm_escalation,
                retrieved,
                freshness,
            )

        selected = retrieved[0]
        answer = selected.text
        model_usage: dict[str, int | str] = {}
        if self.answerer is not None:
            model_response = self.answerer.answer(
                redacted_question,
                [chunk.text for chunk in retrieved],
            )
            answer = model_response.text
            model_usage = {
                "model": model_response.model,
                "input_tokens": model_response.input_tokens,
                "output_tokens": model_response.output_tokens,
                "gate_input_tokens": 0,
                "gate_output_tokens": 0,
            }
        evidence = [chunk.text for chunk in retrieved]
        grounding = grounding_check(answer, evidence)
        judge_response = None
        if grounding.passed and self.answerer is not None:
            judge_response = self.answerer.classify(
                f"Evidence:\n{'\n\n'.join(evidence)}\n\nCandidate answer:\n{answer}"
            )
            model_usage["gate_model"] = judge_response.model
            model_usage["gate_input_tokens"] = judge_response.input_tokens
            model_usage["gate_output_tokens"] = judge_response.output_tokens
        if not grounding.passed or (
            judge_response is not None and judge_response.text.strip().upper() != "PASS"
        ):
            return self._escalate(
                "The retrieved text did not pass the grounding check.",
                redacted_question,
                user_id,
                user_groups,
                pii_detected,
                injection_detected,
                confirm_escalation,
                retrieved,
                freshness,
                unsupported_claims=grounding.unsupported_claims,
                model_usage=model_usage,
                gate_decision=(judge_response.text.strip().upper() if judge_response else "FAIL"),
            )

        pii_notice = " I redacted sensitive-looking information; please avoid sharing passport or case numbers." if pii_detected else ""
        stale_notice = " This source is more than six months old; confirm that it is current with the immigration team." if freshness[0].is_stale else ""
        return AgentResult(
            status="answer",
            message=f"{answer}{pii_notice}{stale_notice}",
            answer=answer,
            citation=format_citation(selected),
            retrieved=retrieved,
            freshness=freshness,
            pii_detected=pii_detected,
            redacted_question=redacted_question,
            injection_detected=injection_detected,
            model_usage=model_usage,
            gate_decision=(judge_response.text.strip().upper() if judge_response else "LOCAL_PASS"),
        )

    def _escalate(
        self,
        reason: str,
        question: str,
        user_id: str,
        user_groups: set[str] | frozenset[str],
        pii_detected: bool,
        injection_detected: bool,
        confirm_escalation: bool,
        retrieved: Sequence[PolicyChunk] = (),
        freshness: Sequence[Freshness] = (),
        unsupported_claims: Sequence[str] = (),
        model_usage: dict[str, int | str] | None = None,
        gate_decision: str = "NOT_RUN",
    ) -> AgentResult:
        permitted = [
            chunk
            for chunk in retrieved
            if chunk.access_group_ids.intersection(user_groups)
        ]
        findings = [format_citation(chunk) for chunk in permitted]
        draft = {"to": "Immigration Attorney Department", "question": question, "findings": findings, "reason": reason}
        escalation_id = None
        if confirm_escalation:
            escalation_id = f"ESC-{len(self.queue.items) + 1:04d}"
            self.queue.add(
                EscalationItem(
                    escalation_id=escalation_id,
                    user_id=user_id,
                    reason=reason,
                    question=question,
                    findings=findings,
                    created_on=self.today,
                )
            )
        if confirm_escalation:
            message = f"I can't answer confidently. Your question was added to the immigration team queue ({escalation_id})."
        else:
            message = f"I can't answer confidently. {reason} I can offer human review and draft a note to the Immigration Attorney Department; confirm before it is added to the team queue."
        if pii_detected:
            message += " Sensitive-looking information was redacted; avoid sharing passport or case numbers."
        return AgentResult(
            status="escalate",
            message=message,
            retrieved=permitted,
            freshness=list(freshness),
            pii_detected=pii_detected,
            redacted_question=question,
            injection_detected=injection_detected,
            escalation_draft=draft,
            escalation_id=escalation_id,
            unsupported_claims=list(unsupported_claims),
            model_usage=model_usage or {},
            gate_decision=gate_decision,
        )