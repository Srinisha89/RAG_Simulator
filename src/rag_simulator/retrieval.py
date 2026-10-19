from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Sequence

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "say",
    "the",
    "to",
    "us",
    "what",
    "when",
    "where",
    "who",
    "why",
    "demo",
    "fictional",
    "synthetic",
}


@dataclass(frozen=True)
class PolicyChunk:
    document_id: str
    title: str
    document_type: str
    section: str
    text: str
    version: str
    last_updated: date
    access_group_ids: frozenset[str]


@dataclass(frozen=True)
class Freshness:
    is_stale: bool
    behind_latest: bool
    loaded_version: str
    latest_version: str | None
    age_days: int


def format_citation(chunk: PolicyChunk) -> str:
    return (
        f"{chunk.title}, {chunk.section}, version {chunk.version}, "
        f"last updated {chunk.last_updated.isoformat()}"
    )


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"\[redacted\]", " ", text, flags=re.I)
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized.lower())


def chunk_documents(documents: Iterable[dict]) -> list[PolicyChunk]:
    """Build one self-contained chunk per structural document rule."""
    chunks = []
    for document in documents:
        sections = document["sections"]
        section_names = " / ".join(section["heading"] for section in sections)
        body = "\n".join(
            f"{section['heading']}: {section['text']}" for section in sections
        )
        chunks.append(
            PolicyChunk(
                document_id=document["document_id"],
                title=document["title"],
                document_type=document["type"],
                section=section_names,
                text=body,
                version=document["version"],
                last_updated=date.fromisoformat(document["last_updated"]),
                access_group_ids=frozenset(document["access_group_ids"]),
            )
        )
    return chunks


def _score(
    query: str,
    chunk: PolicyChunk,
    *,
    document_frequencies: dict[str, int] | None = None,
    corpus_size: int = 1,
) -> float:
    query_tokens = [token for token in _tokenize(query) if token not in STOP_WORDS]
    title_tokens = [token for token in _tokenize(chunk.title) if token not in STOP_WORDS]
    searchable = f"{chunk.section} {chunk.text}"
    document_tokens = [token for token in _tokenize(searchable) if token not in STOP_WORDS]
    if not query_tokens or not document_tokens:
        return 0.0

    query_terms = set(query_tokens)
    counts = {token: document_tokens.count(token) for token in set(document_tokens)}
    frequencies = document_frequencies or {}
    term_weights = {
        token: 1.0 + (corpus_size / (1.0 + frequencies.get(token, 0))) ** 0.5
        for token in query_terms
    }
    total_weight = sum(term_weights.values())
    matched_weight = sum(
        weight
        for token, weight in term_weights.items()
        if counts.get(token, 0) or token in title_tokens
    )
    coverage = matched_weight / total_weight
    title_overlap = sum(
        weight for token, weight in term_weights.items() if token in title_tokens
    ) / total_weight
    phrase_bonus = 1.0 if query.lower() in f"{chunk.title} {searchable}".lower() else 0.0
    return coverage + (0.45 * title_overlap) + phrase_bonus


def search_policy(
    query: str,
    chunks: Sequence[PolicyChunk],
    *,
    user_groups: set[str] | frozenset[str],
    reranker: Callable[[str, list[PolicyChunk]], list[PolicyChunk]] | None = None,
    source_precedence: str | None = None,
    semantic_document_ids: Sequence[str] = (),
    dense_search: Callable[[str, set[str] | frozenset[str], int], Sequence[str]] | None = None,
    candidate_limit: int = 20,
    result_limit: int = 5,
    min_relevance: float = 0.44,
) -> list[PolicyChunk]:
    """Search synthetic policy chunks, filtering permissions before ranking."""
    permitted = [chunk for chunk in chunks if chunk.access_group_ids.intersection(user_groups)]
    corpus_size = len(permitted)
    document_frequencies: dict[str, int] = {}
    for chunk in permitted:
        tokens = set(
            token
            for token in _tokenize(f"{chunk.title} {chunk.section} {chunk.text}")
            if token not in STOP_WORDS
        )
        for token in tokens:
            document_frequencies[token] = document_frequencies.get(token, 0) + 1
    scores = {
        chunk.document_id: _score(
            query,
            chunk,
            document_frequencies=document_frequencies,
            corpus_size=corpus_size,
        )
        for chunk in permitted
    }
    lexical = [
        chunk
        for chunk in sorted(permitted, key=lambda item: scores[item.document_id], reverse=True)
        if scores[chunk.document_id] >= min_relevance
    ][:candidate_limit]
    allowed_ids = {chunk.document_id for chunk in permitted}
    if dense_search is not None:
        semantic_document_ids = dense_search(query, user_groups, candidate_limit)
    semantic_ids = set(semantic_document_ids).intersection(allowed_ids)
    semantic = [chunk for chunk in permitted if chunk.document_id in semantic_ids]
    semantic_by_id = {chunk.document_id: chunk for chunk in semantic}
    candidates_by_id = {chunk.document_id: chunk for chunk in lexical}
    candidates = list(semantic_by_id.values()) + [
        chunk for chunk in candidates_by_id.values() if chunk.document_id not in semantic_by_id
    ]
    candidates = candidates[:candidate_limit]

    if reranker is not None:
        ranked = reranker(query, candidates)
    else:
        ranked = candidates

    for chunk in semantic:
        scores[chunk.document_id] = max(scores.get(chunk.document_id, 0), min_relevance)

    if source_precedence is not None:
        ranked.sort(
            key=lambda chunk: (
                -scores.get(chunk.document_id, 0),
                chunk.document_type != source_precedence,
            )
        )
    if ranked:
        best_score = scores.get(ranked[0].document_id, 0)
        ranked = [
            chunk
            for chunk in ranked
            if scores.get(chunk.document_id, 0) >= max(min_relevance, best_score * 0.85)
        ]
    return ranked[:result_limit]


def check_freshness(
    chunk: PolicyChunk,
    latest_versions: dict[str, str],
    *,
    today: date | None = None,
    max_age_days: int = 180,
) -> Freshness:
    current_date = today or date.today()
    age_days = (current_date - chunk.last_updated).days
    latest_version = latest_versions.get(chunk.document_id)
    return Freshness(
        is_stale=age_days > max_age_days,
        behind_latest=latest_version is not None and latest_version != chunk.version,
        loaded_version=chunk.version,
        latest_version=latest_version,
        age_days=age_days,
    )


class LocalPolicyIndex:
    """Optional local SentenceTransformer + ChromaDB index for dense retrieval."""

    def __init__(self, persist_directory: str, model_name: str):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the retrieval extra to build the local embedding index."
            ) from error

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection("fictional_policy")
        self._model = SentenceTransformer(model_name, local_files_only=True)

    def refresh(self, chunks: Sequence[PolicyChunk]) -> None:
        if self._collection.count():
            self._collection.delete(where={"kind": "policy_chunk"})
        for chunk in chunks:
            metadata = {
                "kind": "policy_chunk",
                "document_id": chunk.document_id,
                "title": chunk.title,
                "document_type": chunk.document_type,
                "section": chunk.section,
                "version": chunk.version,
                "last_updated": chunk.last_updated.isoformat(),
            }
            metadata.update({f"group_{group}": True for group in chunk.access_group_ids})
            self._collection.add(
                ids=[f"{chunk.document_id}:{chunk.section}"],
                documents=[chunk.text],
                metadatas=[metadata],
                embeddings=[self._model.encode(chunk.text).tolist()],
            )

    def search(self, query: str, user_groups: set[str] | frozenset[str], limit: int = 20):
        if not user_groups:
            return []
        permission_filter = {
            "$or": [{f"group_{group}": {"$eq": True}} for group in user_groups]
        }
        response = self._collection.query(
            query_embeddings=[self._model.encode(query).tolist()],
            n_results=limit,
            where=permission_filter,
        )
        return [metadata["document_id"] for metadata in response["metadatas"][0]]