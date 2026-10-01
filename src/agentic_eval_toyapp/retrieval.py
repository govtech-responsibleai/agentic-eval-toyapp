"""Immutable corpus loading and exact FAISS cosine-similarity retrieval."""

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import ValidationError

from agentic_eval_toyapp.embeddings import (
    EmbeddingProvider,
    l2_normalize_embeddings,
    validate_embedding_matrix,
)
from agentic_eval_toyapp.models import (
    DocumentChunk,
    KnowledgeCorpus,
    RetrievedChunk,
    RetrieverMetadata,
)

DEFAULT_CORPUS_PATH = Path(__file__).parent / "resources" / "documents.json"


class CorpusLoadError(ValueError):
    """Raised when a corpus file cannot be read or validated."""


@runtime_checkable
class Retriever(Protocol):
    """Narrow search interface consumed by the RAG agent."""

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        """Return ranked chunks for one non-blank query."""
        ...


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> KnowledgeCorpus:
    """Read and validate one immutable JSON corpus."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CorpusLoadError(f"could not read corpus: {path}") from error
    except json.JSONDecodeError as error:
        raise CorpusLoadError(f"corpus is not valid JSON: {path}") from error

    try:
        return KnowledgeCorpus.model_validate(payload)
    except ValidationError as error:
        raise CorpusLoadError(f"corpus does not match its schema: {path}") from error


def corpus_sha256(corpus: KnowledgeCorpus) -> str:
    """Hash a canonical JSON representation of validated corpus content."""
    canonical_json = json.dumps(
        corpus.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _embedding_text(chunk: DocumentChunk) -> str:
    """Include the document title as searchable semantic context."""
    return f"{chunk.title}\n{chunk.text}"


class FaissRetriever:
    """Read-only exact search over L2-normalised corpus embeddings."""

    def __init__(
        self,
        *,
        corpus: KnowledgeCorpus,
        embedding_provider: EmbeddingProvider,
        default_top_k: int = 3,
        minimum_similarity: float = 0.25,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")
        if not np.isfinite(minimum_similarity) or not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be finite and between -1 and 1")

        self._corpus = corpus
        self._embedding_provider = embedding_provider
        self._default_top_k = default_top_k
        self._minimum_similarity = float(minimum_similarity)

        texts = tuple(_embedding_text(chunk) for chunk in corpus.chunks)
        raw_embeddings = embedding_provider.embed(texts)
        validated_embeddings = validate_embedding_matrix(
            raw_embeddings,
            expected_rows=len(corpus.chunks),
            expected_dimension=embedding_provider.dimension,
        )
        normalised_embeddings = l2_normalize_embeddings(validated_embeddings)

        # Import after embedding inference. Loading FAISS before PyTorch can
        # crash during inference on Apple Silicon because of native runtimes.
        import faiss  # pyright: ignore[reportMissingTypeStubs]

        self._index: Any = faiss.IndexFlatIP(embedding_provider.dimension)
        self._index.add(normalised_embeddings)
        self._metadata = RetrieverMetadata(
            corpus_version=corpus.version,
            corpus_sha256=corpus_sha256(corpus),
            embedding_model_id=embedding_provider.model_id,
            embedding_model_revision=embedding_provider.revision,
            embedding_dimension=embedding_provider.dimension,
            chunk_count=len(corpus.chunks),
        )

    @property
    def metadata(self) -> RetrieverMetadata:
        """Return immutable corpus and index provenance."""
        return self._metadata

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        """Return deterministically ranked chunks above the score threshold."""
        if not query.strip():
            raise ValueError("query must be a non-blank string")
        result_limit = self._default_top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k must be at least 1")

        query_embeddings = self._embedding_provider.embed((query,))
        validated_query = validate_embedding_matrix(
            query_embeddings,
            expected_rows=1,
            expected_dimension=self.metadata.embedding_dimension,
        )
        normalised_query = l2_normalize_embeddings(validated_query)

        # Search the complete small corpus before sorting so equal-score results
        # use chunk_id as a deterministic tie-break even at the top-k boundary.
        scores, row_ids = self._index.search(
            normalised_query,
            len(self._corpus.chunks),
        )
        candidates: list[RetrievedChunk] = []
        for score, row_id in zip(scores[0], row_ids[0], strict=True):
            numeric_score = float(np.clip(score, -1.0, 1.0))
            if row_id < 0 or numeric_score < self._minimum_similarity:
                continue
            chunk = self._corpus.chunks[int(row_id)]
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    similarity_score=numeric_score,
                )
            )

        candidates.sort(key=lambda result: (-result.similarity_score, result.chunk_id))
        return tuple(candidates[:result_limit])
