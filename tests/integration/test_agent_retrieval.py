"""Offline integration test for the bounded agent and FAISS retriever."""

from datetime import date

from agentic_eval_toyapp.agent import RagAgent, load_system_prompt
from agentic_eval_toyapp.embeddings import DeterministicEmbeddingProvider
from agentic_eval_toyapp.models import (
    AgentTurnStatus,
    AnswerAction,
    DocumentChunk,
    DocumentMetadata,
    KnowledgeCorpus,
    SearchAction,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
from agentic_eval_toyapp.retrieval import FaissRetriever


def test_scripted_agent_uses_faiss_evidence_for_grounded_answer() -> None:
    """The provider, agent, and real FAISS index should compose offline."""
    travel = DocumentChunk(
        chunk_id="TRAVEL-001#1",
        document_id="TRAVEL-001",
        title="Travel policy",
        text="Domestic meals are limited to $50 per day.",
        metadata=DocumentMetadata(topic="travel", effective_date=date(2026, 1, 1)),
    )
    leave = DocumentChunk(
        chunk_id="LEAVE-001#1",
        document_id="LEAVE-001",
        title="Leave policy",
        text="Employees receive 18 days of annual leave.",
        metadata=DocumentMetadata(topic="leave", effective_date=date(2026, 1, 1)),
    )
    query = "domestic meal allowance"
    embedder = DeterministicEmbeddingProvider(
        {
            f"{travel.title}\n{travel.text}": [1.0, 0.0],
            f"{leave.title}\n{leave.text}": [0.0, 1.0],
            query: [1.0, 0.0],
        }
    )
    retriever = FaissRetriever(
        corpus=KnowledgeCorpus(version="integration-v1", chunks=(leave, travel)),
        embedding_provider=embedder,
        minimum_similarity=0.5,
    )
    provider = ScriptedGenerationProvider(
        (
            SearchAction(query=query),
            AnswerAction(
                markdown="Domestic meals are limited to $50 per day. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        )
    )
    agent = RagAgent(
        provider=provider,
        retriever=retriever,
        system_prompt=load_system_prompt(),
    )

    result = agent.run("How much can I claim for domestic meals?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.citations == ("TRAVEL-001#1",)
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == ["TRAVEL-001#1"]
