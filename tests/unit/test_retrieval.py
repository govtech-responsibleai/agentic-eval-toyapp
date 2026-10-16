"""Tests for corpus validation and exact FAISS retrieval."""

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from agentic_eval_toyapp.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingValidationError,
)
from agentic_eval_toyapp.models import (
    DocumentChunk,
    DocumentMetadata,
    KnowledgeCorpus,
)
from agentic_eval_toyapp.retrieval import (
    CorpusLoadError,
    FaissRetriever,
    corpus_sha256,
    load_corpus,
)


def _chunk(chunk_id: str, text: str, *, title: str = "Policy") -> DocumentChunk:
    document_id = chunk_id.split("#", maxsplit=1)[0]
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        text=text,
        metadata=DocumentMetadata(topic="testing", effective_date=date(2026, 1, 1)),
    )


def _embedding_text(chunk: DocumentChunk) -> str:
    return f"{chunk.title}\n{chunk.text}"


def _retriever(
    chunks: tuple[DocumentChunk, ...],
    vectors: Mapping[str, Sequence[float]],
    *,
    minimum_similarity: float = -1.0,
    default_top_k: int = 3,
) -> FaissRetriever:
    corpus = KnowledgeCorpus(version="test-v1", chunks=chunks)
    provider = DeterministicEmbeddingProvider(vectors)
    return FaissRetriever(
        corpus=corpus,
        embedding_provider=provider,
        minimum_similarity=minimum_similarity,
        default_top_k=default_top_k,
    )


def test_default_corpus_is_valid_versioned_and_synthetic() -> None:
    """The reviewed corpus must remain inspectable, unique, and fictional."""
    corpus = load_corpus()

    assert corpus.version == "northstar-policies-v2"
    assert len(corpus.chunks) == 40
    assert {chunk.document_id for chunk in corpus.chunks} == {
        "AI-001",
        "CARE-001",
        "DATA-001",
        "EQUIPMENT-001",
        "EXPENSE-001",
        "FLEX-001",
        "LEARNING-001",
        "LEAVE-001",
        "OVERTIME-001",
        "PARENTAL-001",
        "PROCUREMENT-001",
        "SECURITY-001",
        "SICK-001",
        "TRAVEL-001",
        "VENDOR-001",
        "WELLBEING-001",
    }
    assert len({chunk.chunk_id for chunk in corpus.chunks}) == 40
    assert all(
        "@" not in chunk.text or "@northstar.example.test" in chunk.text
        for chunk in corpus.chunks
    )


def test_expanded_corpus_preserves_existing_benchmark_facts() -> None:
    """Expansion must not silently rewrite facts used by MVP regressions."""
    chunks = {chunk.chunk_id: chunk for chunk in load_corpus().chunks}

    assert "18 days of annual leave" in chunks["LEAVE-001#1"].text
    assert "$1,200" in chunks["LEARNING-001#1"].text
    assert "more than $25" in chunks["EXPENSE-001#1"].text
    assert "from $500 through $5,000" in chunks["PROCUREMENT-001#1"].text
    assert "first 60 calendar days" in chunks["FLEX-001#2"].text


def test_default_corpus_checksum_is_stable() -> None:
    """Unexpected corpus edits must be visible to future benchmark fixtures."""
    assert corpus_sha256(load_corpus()) == (
        "6b7fe1276ee59acb85e0e5b539648c000823e35940496cd8f59e0869f48ddc3e"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"version": "v1", "chunks": []},
        {
            "version": "v1",
            "chunks": [
                {
                    "chunk_id": "POLICY-001#1",
                    "document_id": "POLICY-001",
                    "title": "Policy",
                    "text": "One",
                    "metadata": {"topic": "test", "effective_date": "2026-01-01"},
                },
                {
                    "chunk_id": "POLICY-001#1",
                    "document_id": "POLICY-001",
                    "title": "Policy",
                    "text": "Two",
                    "metadata": {"topic": "test", "effective_date": "2026-01-01"},
                },
            ],
        },
        {"version": "v1", "chunks": [{"chunk_id": "incomplete"}]},
    ],
)
def test_load_corpus_rejects_malformed_records(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """Empty, duplicate, and schema-invalid corpora cannot be indexed."""
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusLoadError, match="schema"):
        load_corpus(path)


def test_load_corpus_rejects_invalid_json_and_missing_files(tmp_path: Path) -> None:
    """File and JSON errors have a stable domain exception."""
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")

    with pytest.raises(CorpusLoadError, match="valid JSON"):
        load_corpus(invalid_path)
    with pytest.raises(CorpusLoadError, match="could not read"):
        load_corpus(tmp_path / "missing.json")


def test_faiss_retriever_ranks_filters_and_limits_results() -> None:
    """Normalised inner product produces thresholded cosine ranking."""
    exact = _chunk("POLICY-001#1", "Exact match")
    related = _chunk("POLICY-002#1", "Related match")
    opposite = _chunk("POLICY-003#1", "Opposite match")
    query = "semantic query"
    retriever = _retriever(
        (opposite, related, exact),
        {
            _embedding_text(exact): [1.0, 0.0],
            _embedding_text(related): [0.8, 0.6],
            _embedding_text(opposite): [-1.0, 0.0],
            query: [1.0, 0.0],
        },
        minimum_similarity=0.5,
    )

    results = retriever.search(query, top_k=10)

    assert [result.chunk_id for result in results] == [
        "POLICY-001#1",
        "POLICY-002#1",
    ]
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[1].similarity_score == pytest.approx(0.8)


def test_faiss_retriever_uses_chunk_id_for_equal_score_ties() -> None:
    """Equal scores remain stable regardless of corpus insertion order."""
    chunk_b = _chunk("POLICY-002#1", "Second")
    chunk_a = _chunk("POLICY-001#1", "First")
    query = "same score"
    retriever = _retriever(
        (chunk_b, chunk_a),
        {
            _embedding_text(chunk_a): [1.0, 0.0],
            _embedding_text(chunk_b): [1.0, 0.0],
            query: [1.0, 0.0],
        },
    )

    assert [result.chunk_id for result in retriever.search(query, top_k=2)] == [
        "POLICY-001#1",
        "POLICY-002#1",
    ]


def test_faiss_retriever_supports_unicode() -> None:
    """Unicode corpus text and queries pass through without lossy conversion."""
    chunk = _chunk("POLICY-001#1", "Café reimbursement in 日本")
    query = "日本のカフェ"
    retriever = _retriever(
        (chunk,),
        {_embedding_text(chunk): [1.0, 1.0], query: [1.0, 1.0]},
    )

    assert retriever.search(query)[0].text == "Café reimbursement in 日本"


@pytest.mark.parametrize("query", ["", "   "])
def test_faiss_retriever_rejects_blank_queries(query: str) -> None:
    """Blank input has no meaningful semantic representation."""
    chunk = _chunk("POLICY-001#1", "Policy text")
    retriever = _retriever((chunk,), {_embedding_text(chunk): [1.0]})

    with pytest.raises(ValueError, match="query"):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_faiss_retriever_rejects_invalid_top_k(top_k: int) -> None:
    """Retrieval must request at least one possible result."""
    chunk = _chunk("POLICY-001#1", "Policy text")
    query = "query"
    vectors = {_embedding_text(chunk): [1.0], query: [1.0]}
    retriever = _retriever((chunk,), vectors)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search(query, top_k=top_k)
    with pytest.raises(ValueError, match="default_top_k"):
        _retriever((chunk,), vectors, default_top_k=top_k)


@pytest.mark.parametrize("threshold", [-1.01, 1.01, float("nan")])
def test_faiss_retriever_rejects_invalid_threshold(threshold: float) -> None:
    """Cosine thresholds must be finite and in the cosine range."""
    chunk = _chunk("POLICY-001#1", "Policy text")

    with pytest.raises(ValueError, match="minimum_similarity"):
        _retriever(
            (chunk,),
            {_embedding_text(chunk): [1.0]},
            minimum_similarity=threshold,
        )


class _WrongDimensionProvider:
    model_id = "broken"
    revision = "v1"
    dimension = 2

    def __init__(self) -> None:
        self._calls = 0

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self._calls += 1
        if self._calls == 1:
            return np.ones((len(texts), 2), dtype=np.float32)
        return np.ones((len(texts), 3), dtype=np.float32)


def test_faiss_retriever_rejects_query_dimension_drift() -> None:
    """A provider cannot change embedding dimensions after index creation."""
    corpus = KnowledgeCorpus(
        version="test-v1",
        chunks=(_chunk("POLICY-001#1", "Policy text"),),
    )
    retriever = FaissRetriever(
        corpus=corpus,
        embedding_provider=_WrongDimensionProvider(),
        minimum_similarity=-1.0,
    )

    with pytest.raises(EmbeddingValidationError, match="dimension"):
        retriever.search("query")


def test_faiss_retriever_exposes_reproducibility_metadata() -> None:
    """Every index identifies its corpus, model fixture, and dimensions."""
    chunk = _chunk("POLICY-001#1", "Policy text")
    retriever = _retriever((chunk,), {_embedding_text(chunk): [1.0, 0.0]})

    assert retriever.metadata.corpus_version == "test-v1"
    assert retriever.metadata.embedding_model_id == "deterministic-embedding-fixture"
    assert retriever.metadata.embedding_model_revision == "v1"
    assert retriever.metadata.embedding_dimension == 2
    assert retriever.metadata.index_type == "IndexFlatIP"
    assert retriever.metadata.chunk_count == 1
