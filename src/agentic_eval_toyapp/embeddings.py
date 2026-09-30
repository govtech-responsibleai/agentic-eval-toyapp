"""Embedding providers and validation for reproducible semantic retrieval."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float32]


class EmbeddingValidationError(ValueError):
    """Raised when an embedding provider returns an invalid matrix."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Narrow interface shared by real and deterministic embedders."""

    @property
    def model_id(self) -> str:
        """Return the stable embedding model identifier."""
        ...

    @property
    def revision(self) -> str:
        """Return the immutable model or fixture revision."""
        ...

    @property
    def dimension(self) -> int:
        """Return the number of values in each embedding."""
        ...

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed a non-empty sequence of non-blank texts."""
        ...


def validate_embedding_matrix(
    embeddings: ArrayLike,
    *,
    expected_rows: int | None = None,
    expected_dimension: int | None = None,
) -> FloatMatrix:
    """Convert provider output to a finite, non-zero float32 matrix."""
    try:
        matrix = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise EmbeddingValidationError("embeddings must be numeric") from error

    if matrix.ndim != 2:
        raise EmbeddingValidationError("embeddings must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise EmbeddingValidationError("embeddings must not be empty")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise EmbeddingValidationError(
            f"expected {expected_rows} embedding rows, got {matrix.shape[0]}"
        )
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise EmbeddingValidationError(
            f"expected embedding dimension {expected_dimension}, got {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise EmbeddingValidationError("embeddings must contain only finite values")

    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all():
        raise EmbeddingValidationError("embedding norms must be finite")
    if np.any(norms == 0.0):
        raise EmbeddingValidationError("embeddings must not contain zero vectors")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def l2_normalize_embeddings(embeddings: ArrayLike) -> FloatMatrix:
    """Return row-wise L2-normalised embeddings suitable for cosine search."""
    matrix = validate_embedding_matrix(embeddings)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
    """Validate provider input without silently changing user text."""
    values = tuple(texts)
    if not values:
        raise ValueError("at least one text is required")
    if any(not value.strip() for value in values):
        raise ValueError("embedding texts must be non-blank strings")
    return values


class DeterministicEmbeddingProvider:
    """Return fixed vectors for exact, network-free tests."""

    def __init__(
        self,
        vectors_by_text: Mapping[str, Sequence[float]],
        *,
        model_id: str = "deterministic-embedding-fixture",
        revision: str = "v1",
    ) -> None:
        if not model_id.strip() or not revision.strip():
            raise ValueError("model_id and revision must not be blank")
        if not vectors_by_text:
            raise ValueError("at least one deterministic vector is required")

        validated: dict[str, FloatMatrix] = {}
        dimension: int | None = None
        for text, vector in vectors_by_text.items():
            if not text.strip():
                raise ValueError("deterministic vector keys must be non-blank strings")
            matrix = validate_embedding_matrix([vector])
            if dimension is None:
                dimension = int(matrix.shape[1])
            elif matrix.shape[1] != dimension:
                raise EmbeddingValidationError(
                    "all deterministic vectors must have the same dimension"
                )
            validated[text] = matrix

        if (
            dimension is None
        ):  # Defensive; the mapping is already known to be non-empty.
            raise ValueError("at least one deterministic vector is required")
        self._vectors_by_text = validated
        self._model_id = model_id.strip()
        self._revision = revision.strip()
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        values = _validate_texts(texts)
        missing = [text for text in values if text not in self._vectors_by_text]
        if missing:
            raise ValueError(f"no deterministic embedding configured for: {missing!r}")
        return np.vstack([self._vectors_by_text[text] for text in values]).astype(
            np.float32,
            copy=False,
        )


class SentenceTransformerEmbeddingProvider:
    """Pinned SentenceTransformer adapter for live semantic retrieval."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not model_id.strip() or not revision.strip() or not device.strip():
            raise ValueError("model_id, revision, and device must not be blank")

        from sentence_transformers import SentenceTransformer

        self._model_id = model_id.strip()
        self._revision = revision.strip()
        self._model = SentenceTransformer(
            self._model_id,
            revision=self._revision,
            device=device.strip(),
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise EmbeddingValidationError(
                "SentenceTransformer did not report a valid embedding dimension"
            )
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        values = _validate_texts(texts)
        encoded = self._model.encode(  # pyright: ignore[reportUnknownMemberType]
            list(values),
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return validate_embedding_matrix(
            encoded,
            expected_rows=len(values),
            expected_dimension=self.dimension,
        )
