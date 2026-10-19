from datetime import date

from rag_simulator.retrieval import (
    chunk_documents,
    check_freshness,
    format_citation,
    search_policy,
)


def test_structural_chunking_keeps_exception_with_its_rule(policy_documents):
    chunks = chunk_documents(policy_documents)

    extension_chunk = next(chunk for chunk in chunks if chunk.document_id == "POL-001")
    assert "120 days" in extension_chunk.text
    assert "Unless the immigration team" in extension_chunk.text
    assert "Exception" in extension_chunk.section


def test_permission_filter_runs_before_reranking(policy_documents):
    seen_by_reranker = []

    def recording_reranker(query, candidates):
        seen_by_reranker.extend(candidates)
        return candidates

    chunks = chunk_documents(policy_documents)
    results = search_policy(
        "internal case notes",
        chunks,
        user_groups={"employees"},
        reranker=recording_reranker,
    )

    assert all("legal-ops" not in candidate.access_group_ids for candidate in seen_by_reranker)
    assert all(result.document_id != "POL-004" for result in results)


def test_hybrid_search_finds_exact_form_identifier(policy_documents):
    documents = policy_documents + [
        {
            "document_id": "POL-021",
            "title": "FICTIONAL: Form I-129 Field Guide",
            "type": "company_policy",
            "sections": [{"heading": "Part 1", "text": "Use the fictional field guide for this example."}],
            "version": "1.0",
            "last_updated": "2026-08-01",
            "access_group_ids": ["employees"],
        }
    ]
    chunks = chunk_documents(documents)

    results = search_policy("I-129", chunks, user_groups={"employees"})

    assert results
    assert results[0].document_id == "POL-021"


def test_specific_form_field_question_survives_relevance_threshold(policy_documents):
    results = search_policy(
        "How do I ask the fictional company for help with a form field?",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert results
    assert results[0].document_id == "POL-003"


def test_low_confidence_tail_is_removed_from_extension_window_results(policy_documents):
    results = search_policy(
        "What is the fictional extension request window?",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert [result.document_id for result in results] == ["POL-001"]


def test_title_match_breaks_ties_against_generic_body_overlap(policy_documents):
    results = search_policy(
        "What is the fictional travel review process?",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert results[0].document_id == "POL-002"
    assert "POL-015" not in {result.document_id for result in results}


def test_travel_checklist_query_ranks_its_stale_document_first(policy_documents):
    results = search_policy(
        "What does the fictional travel document checklist say?",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert results
    assert results[0].document_id == "POL-008"


def test_redaction_marker_does_not_block_retrieval(policy_documents):
    results = search_policy(
        "What does this fictional form field mean? [REDACTED]",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert results
    assert results[0].document_id == "POL-003"


def test_hybrid_search_includes_semantic_matches_without_keyword_overlap(policy_documents):
    chunks = chunk_documents(policy_documents)

    results = search_policy(
        "vacation plans",
        chunks,
        user_groups={"employees"},
        semantic_document_ids=["POL-002"],
    )

    assert results
    assert results[0].document_id == "POL-002"


def test_retrieval_ranks_specific_terms_above_english_stopwords(policy_documents):
    results = search_policy(
        "Can my dependent work in the US?",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert results
    assert results[0].document_id == "POL-006"


def test_internal_engineering_travel_query_prefers_company_policy(policy_documents):
    results = search_policy(
        "What does the fictional engineering travel approval document say?",
        chunk_documents(policy_documents),
        user_groups={"employees", "engineering"},
        source_precedence="company_policy",
    )

    assert results
    assert results[0].document_id == "POL-019"


def test_search_does_not_reveal_restricted_documents_in_results(policy_documents):
    chunks = chunk_documents(policy_documents)

    results = search_policy("case notes", chunks, user_groups={"employees"})

    assert "POL-004" not in {result.document_id for result in results}


def test_dense_search_receives_user_groups_and_cannot_return_forbidden_chunks(policy_documents):
    observed_groups = []

    def dense_search(query, user_groups, limit):
        observed_groups.append(user_groups)
        return ["POL-004", "POL-002"]

    results = search_policy(
        "case notes",
        chunk_documents(policy_documents),
        user_groups={"employees"},
        dense_search=dense_search,
    )

    assert observed_groups == [{"employees"}]
    assert "POL-004" not in {result.document_id for result in results}


def test_freshness_flags_stale_and_unloaded_latest_versions(policy_documents):
    chunks = chunk_documents(policy_documents)
    stale_trap = next(chunk for chunk in chunks if chunk.document_id == "POL-016")
    old_document = next(chunk for chunk in chunks if chunk.document_id == "POL-004")

    lag = check_freshness(stale_trap, {"POL-016": "2.0"}, today=date(2026, 10, 19))
    stale = check_freshness(old_document, {}, today=date(2026, 10, 19))

    assert lag.behind_latest is True
    assert lag.loaded_version == "1.0"
    assert lag.latest_version == "2.0"
    assert stale.is_stale is True


def test_company_policy_ranks_first_for_internal_process_queries(policy_documents):
    chunks = chunk_documents(policy_documents)

    results = search_policy(
        "internal process travel review",
        chunks,
        user_groups={"employees"},
        source_precedence="company_policy",
    )

    assert results
    assert results[0].document_type == "company_policy"


def test_freshness_window_is_configurable(policy_documents):
    chunk = chunk_documents(policy_documents)[0]

    freshness = check_freshness(
        chunk,
        {},
        today=date(2026, 10, 19),
        max_age_days=365,
    )

    assert freshness.is_stale is False


def test_citation_includes_document_section_version_and_update_date(policy_documents):
    chunk = next(chunk for chunk in chunk_documents(policy_documents) if chunk.document_id == "POL-001")

    citation = format_citation(chunk)

    assert "FICTIONAL: Extension Request Workflow" in citation
    assert "Request window" in citation
    assert "Exception" in citation
    assert "version 2.0" in citation
    assert "2026-08-14" in citation
