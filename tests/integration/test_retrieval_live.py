"""Opt-in checks for the pinned SentenceTransformer and FAISS index."""

from collections.abc import Iterator

import anyio
import pytest
from anyio import CapacityLimiter

from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.embeddings import SentenceTransformerEmbeddingProvider
from agentic_eval_toyapp.models import RetrievedChunk
from agentic_eval_toyapp.retrieval import AsyncRetriever, FaissRetriever, load_corpus


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
        (
            "I have been ill for three workdays. What proof do I need?",
            "SICK-001",
        ),
        (
            "Can I use the remote-work allowance for my monthly internet bill?",
            "EQUIPMENT-001",
        ),
        (
            "I need to take my parent to a medical appointment. Which leave applies?",
            "CARE-001",
        ),
        (
            "How much leave does a supporting caregiver receive for a new child?",
            "PARENTAL-001",
        ),
        (
            "Does time fixing an incident while on call count as overtime?",
            "OVERTIME-001",
        ),
        (
            "Can I carry unused sick leave into next year?",
            "SICK-001",
        ),
        (
            "Can a supplier accessing company systems skip review for a cheap order?",
            "VENDOR-001",
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


@pytest.mark.live
@pytest.mark.parametrize(
    ("query", "expected_chunk_ids"),
    [
        (
            "Can I paste Restricted information into an approved AI assistant?",
            {"AI-001#1", "DATA-001#2"},
        ),
        (
            "Does a security review replace two quotations for a $900 supplier?",
            {"PROCUREMENT-001#1", "VENDOR-001#2"},
        ),
        (
            "Can I use my learning budget for a gym membership?",
            {"LEARNING-001#1", "WELLBEING-001#1"},
        ),
    ],
)
def test_pinned_model_retrieves_overlapping_policy_evidence(
    live_retriever: FaissRetriever,
    query: str,
    expected_chunk_ids: set[str],
) -> None:
    """Top-three candidates should retain both sides of reviewed policy overlaps."""
    results = live_retriever.search(query, top_k=3)

    assert expected_chunk_ids <= {result.chunk_id for result in results}


@pytest.mark.live
@pytest.mark.anyio
async def test_fifty_concurrent_real_searches_match_serial_results(
    live_retriever: FaissRetriever,
) -> None:
    """The single slot must serialise real searches without errors or drift."""
    queries = [
        "Can someone who just joined work away from the office?",
        "Do I need proof for a forty-dollar taxi ride?",
        "Can I buy a nine-hundred-dollar item directly?",
        "What should I do if I think my account has been compromised?",
        "May I paste sensitive information into a public chatbot?",
    ] * 10
    serial = [live_retriever.search(query) for query in queries]
    adapter = AsyncRetriever(live_retriever, CapacityLimiter(1))
    concurrent: list[tuple[RetrievedChunk, ...] | None] = [None] * len(queries)

    async def run(index: int) -> None:
        concurrent[index] = await adapter.search(queries[index])

    async with anyio.create_task_group() as group:
        for index in range(len(queries)):
            group.start_soon(run, index)

    assert concurrent == serial
