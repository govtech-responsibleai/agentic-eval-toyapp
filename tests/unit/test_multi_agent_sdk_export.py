"""Verify real SDK span lifecycle without network or Langfuse server timing."""

from collections import Counter
from datetime import date
from uuid import uuid4

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
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
    AnswerAction,
    DocumentMetadata,
    ReportMediaType,
    ReportUploadResponse,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.multi_agent_tracing import LangfuseCoordinatorTraceRecorder
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorAnswerAction,
    CoordinatorRouteAnalysisAction,
    CoordinatorRouteRagAction,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
from agentic_eval_toyapp.reports import StoredReport


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


def test_two_direct_turns_are_exported_as_two_complete_sdk_trees() -> None:
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
        assert agent.run("First message", session_id="test-session").message == (
            "First reply."
        )
        assert agent.run("Second message", session_id="test-session").message == (
            "Second reply."
        )
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


def test_rag_route_exports_one_complete_sdk_tree() -> None:
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
        retriever=_Retriever(),
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
        agent.run("How much annual leave?", session_id="test-rag-session")
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


def test_analysis_route_exports_one_complete_sdk_tree() -> None:
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
        agent.run(
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
