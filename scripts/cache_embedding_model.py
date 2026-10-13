"""Download and validate the pinned embedding model during an image build."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the image-build utility."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def cache_model(*, model_id: str, revision: str, cache_dir: Path) -> int:
    """Cache one immutable SentenceTransformer revision and return its dimension."""
    stripped_model_id = model_id.strip()
    stripped_revision = revision.strip()
    if not stripped_model_id or not stripped_revision:
        raise ValueError("model_id and revision must not be blank")

    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        stripped_model_id,
        revision=stripped_revision,
        device="cpu",
        cache_folder=str(cache_dir),
        trust_remote_code=False,
    )
    dimension = model.get_embedding_dimension()
    if dimension is None or dimension <= 0:
        raise RuntimeError("cached model did not report a valid embedding dimension")
    return dimension


def main() -> None:
    """Cache the requested model or fail the image build."""
    arguments = build_parser().parse_args()
    cache_model(
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
    )


if __name__ == "__main__":
    main()
