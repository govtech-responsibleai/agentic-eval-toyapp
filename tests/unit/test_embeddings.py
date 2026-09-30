"""Tests for embedding validation and deterministic providers."""

import numpy as np
import pytest

from agentic_eval_toyapp.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingValidationError,
    l2_normalize_embeddings,
    validate_embedding_matrix,
)


def test_validate_embedding_matrix_returns_contiguous_float32() -> None:
    """FAISS receives a predictable numeric representation."""
    matrix = validate_embedding_matrix([[3, 4], [1, 2]])

    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize(
    "embeddings",
    [
        [],
        [1.0, 2.0],
        [[]],
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
        [["not-a-number", 1.0]],
    ],
)
def test_validate_embedding_matrix_rejects_invalid_values(
    embeddings: object,
) -> None:
    """Empty, malformed, zero, and non-finite vectors cannot enter an index."""
    with pytest.raises(EmbeddingValidationError):
        validate_embedding_matrix(embeddings)  # type: ignore[arg-type]


def test_validate_embedding_matrix_checks_shape_contract() -> None:
    """Provider row and dimension drift must fail before FAISS is called."""
    with pytest.raises(EmbeddingValidationError, match="rows"):
        validate_embedding_matrix([[1.0, 0.0]], expected_rows=2)
    with pytest.raises(EmbeddingValidationError, match="dimension"):
        validate_embedding_matrix([[1.0, 0.0]], expected_dimension=3)


def test_l2_normalize_embeddings_produces_unit_vectors() -> None:
    """Inner product over unit vectors is cosine similarity."""
    normalised = l2_normalize_embeddings([[3.0, 4.0], [0.0, 2.0]])

    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), [1.0, 1.0])


def test_deterministic_provider_returns_exact_vectors() -> None:
    """Offline tests can control retrieval rankings without a model download."""
    provider = DeterministicEmbeddingProvider({"alpha": [1.0, 0.0], "βeta": [0.0, 2.0]})

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 2
    np.testing.assert_array_equal(
        provider.embed(("βeta", "alpha")),
        np.asarray([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32),
    )


@pytest.mark.parametrize("texts", [(), ("",), ("   ",)])
def test_deterministic_provider_rejects_empty_text(texts: tuple[str, ...]) -> None:
    """Blank queries or documents are invalid embedding inputs."""
    provider = DeterministicEmbeddingProvider({"known": [1.0]})

    with pytest.raises(ValueError):
        provider.embed(texts)


def test_deterministic_provider_rejects_unknown_text() -> None:
    """Missing fixtures must fail explicitly instead of returning random data."""
    provider = DeterministicEmbeddingProvider({"known": [1.0]})

    with pytest.raises(ValueError, match="unknown"):
        provider.embed(("unknown",))


def test_deterministic_provider_rejects_mismatched_dimensions() -> None:
    """Every vector from one provider must share one fixed dimension."""
    with pytest.raises(EmbeddingValidationError, match="same dimension"):
        DeterministicEmbeddingProvider(
            {"two dimensions": [1.0, 0.0], "three dimensions": [1.0, 0.0, 0.0]}
        )
