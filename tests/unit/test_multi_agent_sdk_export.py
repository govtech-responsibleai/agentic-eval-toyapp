"""Verify real SDK span lifecycle without network or Langfuse server timing."""

import asyncio
import threading
from collections import Counter
from datetime import date
from uuid import uuid4

import pytest
from anyio import CapacityLimiter
from langfuse import Langfuse
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from agentic_eval_toyapp.agent import RagAgent, RagSpecialistAdapter
from agentic_eval_toyapp.analysis import (
    DeterministicAnalysisGenerationProvider,
    ReportAnalysisAgent,
)
from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    ScriptedCoordinatorGenerationProvider,
)
from agentic_eval_toyapp.models import (
    AgentAction,
    AnswerAction,
    DocumentMetadata,
    ModelRequest,
    ReportMediaType,
    ReportUploadResponse,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.multi_agent_tracing import (
    LangfuseCoordinatorTraceRecorder,
    delegated_rag_trace,
)
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorAction,
    CoordinatorAnswerAction,
    CoordinatorRequest,
    CoordinatorRouteAnalysisAction,
    CoordinatorRouteRagAction,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.retrieval import AsyncRetriever


def _recorder() -> tuple[LangfuseCoordinatorTraceRecorder, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    public_key = f"pk-lf-test-{uuid4()}"
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-lf-test",
        base_url="https://langfuse.invalid",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    return (
        LangfuseCoordinatorTraceRecorder(
            public_key=public_key,
            secret_key="sk-lf-test",
            base_url="https://langfuse.invalid",
            environment="test",
            release="test",
            base_metadata={},
            rag_model_id="scripted",
            client=client,
        ),
        exporter,
    )


class _Retriever:
    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        del query, top_k
        return (
            RetrievedChunk(
                chunk_id="LEAVE-001#1",
                document_id="LEAVE-001",
                title="Leave Policy",
                text="Employees receive 18 days of annual leave.",
                metadata=DocumentMetadata(
                    topic="leave", effective_date=date(2026, 1, 1)
                ),
                similarity_score=0.91,
            ),
        )


@pytest.mark.anyio
async def test_two_direct_turns_are_exported_as_two_complete_sdk_trees() -> None:
    """A missing live trace must not be caused by an unended app span."""
    recorder, exporter = _recorder()
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (
                CoordinatorAnswerAction(markdown="First reply."),
                CoordinatorAnswerAction(markdown="Second reply."),
            )
        ),
        system_prompt="Coordinate.",
        trace_recorder=recorder,
    )

    try:
        assert (
            await agent.run("First message", session_id="test-session")
        ).message == "First reply."
        assert (
            await agent.run("Second message", session_id="test-session")
        ).message == "Second reply."
        spans = exporter.get_finished_spans()
        assert len(spans) == 6
        by_trace: dict[int, Counter[str]] = {}
        for span in spans:
            assert span.context is not None
            by_trace.setdefault(span.context.trace_id, Counter())[span.name] += 1
        assert len(by_trace) == 2
        assert all(
            names
            == Counter({"agent-turn": 1, "inspect-input": 1, "coordinator-action": 1})
            for names in by_trace.values()
        )
    finally:
        agent.shutdown()


@pytest.mark.anyio
async def test_rag_route_exports_one_complete_sdk_tree() -> None:
    """The real SDK must emit the selected specialist and every tool boundary."""
    recorder, exporter = _recorder()
    rag = RagAgent(
        provider=ScriptedGenerationProvider(
            (
                SearchAction(query="annual leave", top_k=3),
                AnswerAction(
                    markdown="Employees receive 18 days. [LEAVE-001#1]",
                    citations=("LEAVE-001#1",),
                ),
            )
        ),
        retriever=AsyncRetriever(_Retriever(), CapacityLimiter(1)),
        system_prompt="Answer policy questions.",
    )
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorRouteRagAction(task="Explain annual leave"),)
        ),
        system_prompt="Coordinate.",
        rag_specialist=RagSpecialistAdapter(rag),
        trace_recorder=recorder,
    )
    try:
        await agent.run("How much annual leave?", session_id="test-rag-session")
        spans = exporter.get_finished_spans()
        assert Counter(span.name for span in spans) == Counter(
            {
                "agent-turn": 1,
                "inspect-input": 1,
                "coordinator-action": 1,
                "delegate-rag": 1,
                "model-action": 2,
                "search_knowledge_base": 1,
                "validate_answer": 1,
                "return-response": 1,
            }
        )
        assert len({span.context.trace_id for span in spans if span.context}) == 1
        parent_by_name = {
            "inspect-input": "agent-turn",
            "coordinator-action": "agent-turn",
            "delegate-rag": "agent-turn",
            "model-action": "delegate-rag",
            "search_knowledge_base": "delegate-rag",
            "validate_answer": "delegate-rag",
            "return-response": "agent-turn",
        }
        name_by_id = {span.context.span_id: span.name for span in spans if span.context}
        for span in spans:
            if span.name == "agent-turn":
                assert span.parent is None
            else:
                assert span.parent is not None
                assert name_by_id[span.parent.span_id] == parent_by_name[span.name]
    finally:
        agent.shutdown()


@pytest.mark.anyio
async def test_analysis_route_exports_one_complete_sdk_tree() -> None:
    """Report inspection, generation, and validation remain in one trace."""
    recorder, exporter = _recorder()
    text = "# Findings\nRevenue increased by 12%."
    report = StoredReport(
        metadata=ReportUploadResponse(
            file_id=uuid4(),
            filename="findings.md",
            media_type=ReportMediaType.MARKDOWN,
            byte_size=len(text.encode()),
            extracted_text_length=len(text),
            sha256="0" * 64,
        ),
        extracted_text=text,
    )
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (
                CoordinatorRouteAnalysisAction(
                    task="Summarise findings",
                    file_ids=(report.metadata.file_id,),
                ),
            )
        ),
        system_prompt="Coordinate.",
        analysis_specialist=ReportAnalysisAgent(
            provider=DeterministicAnalysisGenerationProvider(),
            system_prompt="Analyse reports.",
        ),
        trace_recorder=recorder,
    )
    try:
        await agent.run(
            "Summarise findings", session_id="test-analysis-session", reports=(report,)
        )
        spans = exporter.get_finished_spans()
        assert Counter(span.name for span in spans) == Counter(
            {
                "agent-turn": 1,
                "inspect-input": 1,
                "coordinator-action": 1,
                "delegate-analysis": 1,
                "inspect-content": 1,
                "analyse-report": 1,
                "validate-report-references": 1,
                "return-response": 1,
            }
        )
        assert len({span.context.trace_id for span in spans if span.context}) == 1
        name_by_id = {span.context.span_id: span.name for span in spans if span.context}
        for span in spans:
            if span.name == "agent-turn":
                assert span.parent is None
            else:
                assert span.parent is not None
                expected_parent = (
                    "delegate-analysis"
                    if span.name
                    in {
                        "inspect-content",
                        "analyse-report",
                        "validate-report-references",
                    }
                    else "agent-turn"
                )
                assert name_by_id[span.parent.span_id] == expected_parent
    finally:
        agent.shutdown()


class _PerMessageRagRouter:
    """Route each coordinator turn to RAG using that turn's own user message."""

    model_id = "per-message-router-fixture"
    revision = "v1"

    async def generate_action(self, request: CoordinatorRequest) -> CoordinatorAction:
        return CoordinatorRouteRagAction(task=request.user_message)


class _GatedRagProvider:
    """Hold every RAG model call open so two turns overlap before finishing."""

    model_id = "gated-rag-fixture"
    revision = "v1"

    def __init__(self, expected_concurrent: int) -> None:
        self._expected = expected_concurrent
        self._arrived = 0
        self._all_arrived = asyncio.Event()
        self._by_query: dict[str, int] = {}

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        if request.model_call_number == 1:
            self._arrived += 1
            if self._arrived == self._expected:
                self._all_arrived.set()
            await asyncio.wait_for(self._all_arrived.wait(), 5.0)
            return SearchAction(query=request.user_message, top_k=3)
        return AnswerAction(
            markdown=f"Answer for {request.user_message}. [LEAVE-001#1]",
            citations=("LEAVE-001#1",),
        )


@pytest.mark.anyio
async def test_concurrent_traced_turns_keep_separate_span_trees() -> None:
    """Overlapping turns must not leak parents across tasks or the thread hop."""
    recorder, exporter = _recorder()
    turn_count = 3
    agent = CoordinatorAgent(
        provider=_PerMessageRagRouter(),
        system_prompt="Coordinate.",
        rag_specialist=RagSpecialistAdapter(
            RagAgent(
                provider=_GatedRagProvider(turn_count),
                retriever=AsyncRetriever(_Retriever(), CapacityLimiter(1)),
                system_prompt="Answer policy questions.",
            )
        ),
        trace_recorder=recorder,
    )
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    agent.run(f"question-{index}", session_id=f"session-{index}")
                    for index in range(turn_count)
                )
            ),
            10.0,
        )
        assert [r.message.split(".")[0] for r in results] == [
            f"Answer for question-{index}" for index in range(turn_count)
        ]
        spans = exporter.get_finished_spans()
        by_trace: dict[int, list[ReadableSpan]] = {}
        for span in spans:
            assert span.context is not None
            by_trace.setdefault(span.context.trace_id, []).append(span)

        assert len(by_trace) == turn_count
        expected = Counter(
            {
                "agent-turn": 1,
                "inspect-input": 1,
                "coordinator-action": 1,
                "delegate-rag": 1,
                "model-action": 2,
                "search_knowledge_base": 1,
                "validate_answer": 1,
                "return-response": 1,
            }
        )
        sessions: set[object] = set()
        for trace_spans in by_trace.values():
            assert Counter(span.name for span in trace_spans) == expected
            ids = {span.context.span_id for span in trace_spans if span.context}
            roots = [span for span in trace_spans if span.parent is None]
            assert [root.name for root in roots] == ["agent-turn"]
            # Every non-root span must hang off a span in its own trace.
            assert all(
                span.parent is None or span.parent.span_id in ids
                for span in trace_spans
            )
            attributes = roots[0].attributes or {}
            sessions.add(attributes.get("session.id"))
            search = next(s for s in trace_spans if s.name == "search_knowledge_base")
            delegate = next(s for s in trace_spans if s.name == "delegate-rag")
            assert search.parent is not None
            assert search.parent.span_id == delegate.context.span_id  # type: ignore[union-attr]
        assert sessions == {f"session-{index}" for index in range(turn_count)}
    finally:
        agent.shutdown()


class _ThreadProbeRetriever(_Retriever):
    """Record the tracing state visible from inside the retrieval worker thread."""

    def __init__(self) -> None:
        self.thread_id: int | None = None
        self.span_id: int | None = None
        self.delegated_trace: object | None = None

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        self.thread_id = threading.get_ident()
        self.span_id = otel_trace.get_current_span().get_span_context().span_id
        self.delegated_trace = delegated_rag_trace()
        return super().search(query, top_k=top_k)


@pytest.mark.anyio
async def test_trace_context_survives_the_retrieval_thread_hop() -> None:
    """The worker thread must see the delegate-rag span as its current parent."""
    recorder, exporter = _recorder()
    retriever = _ThreadProbeRetriever()
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorRouteRagAction(task="Explain annual leave"),)
        ),
        system_prompt="Coordinate.",
        rag_specialist=RagSpecialistAdapter(
            RagAgent(
                provider=ScriptedGenerationProvider(
                    (
                        SearchAction(query="annual leave", top_k=3),
                        AnswerAction(
                            markdown="Employees receive 18 days. [LEAVE-001#1]",
                            citations=("LEAVE-001#1",),
                        ),
                    )
                ),
                retriever=AsyncRetriever(retriever, CapacityLimiter(1)),
                system_prompt="Answer policy questions.",
            )
        ),
        trace_recorder=recorder,
    )
    try:
        await agent.run("How much annual leave?", session_id="thread-hop-session")
        delegate = next(
            span
            for span in exporter.get_finished_spans()
            if span.name == "delegate-rag"
        )
        assert delegate.context is not None
        assert retriever.thread_id is not None
        assert retriever.thread_id != threading.get_ident()
        assert retriever.delegated_trace is not None
        assert retriever.span_id == delegate.context.span_id
    finally:
        agent.shutdown()
