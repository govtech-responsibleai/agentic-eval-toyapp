"""Opt-in checks for the pinned SentenceTransformer and FAISS index."""

from collections.abc import Iterator

import pytest

from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.embeddings import SentenceTransformerEmbeddingProvider
from agentic_eval_toyapp.retrieval import FaissRetriever, load_corpus


@pytest.fixture(scope="module")
def live_retriever() -> Iterator[FaissRetriever]:
    """Download/load the pinned model and build the real corpus index once."""
    settings = AppSettings()
    provider = SentenceTransformerEmbeddingProvider(
        model_id=settings.embedding_model_id,
        revision=settings.embedding_model_revision,
        device=settings.embedding_device,
        cache_dir=settings.embedding_cache_dir,
    )
    yield FaissRetriever(
        corpus=load_corpus(),
        embedding_provider=provider,
        default_top_k=settings.retrieval_top_k,
        minimum_similarity=settings.retrieval_min_similarity,
    )


@pytest.mark.live
@pytest.mark.parametrize(
    ("query", "expected_document_id"),
    [
        ("Can someone who just joined work away from the office?", "FLEX-001"),
        ("Do I need proof for a forty-dollar taxi ride?", "EXPENSE-001"),
        ("Can I buy a nine-hundred-dollar item directly?", "PROCUREMENT-001"),
        (
            "What should I do if I think my account has been compromised?",
            "SECURITY-001",
        ),
        ("May I paste sensitive information into a public chatbot?", "DATA-001"),
        (
            "Does the organisation pay for a second certification attempt?",
            "LEARNING-001",
        ),
    ],
)
def test_pinned_model_retrieves_expected_policy(
    live_retriever: FaissRetriever,
    query: str,
    expected_document_id: str,
) -> None:
    """Natural paraphrases should rank the reviewed policy first."""
    results = live_retriever.search(query, top_k=1)

    assert results
    assert results[0].document_id == expected_document_id
