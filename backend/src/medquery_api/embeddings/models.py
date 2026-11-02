"""The four embedding models from the research (notebooks 04 and 06)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    name: str  # the short name the research used
    model_id: str  # the Hugging Face model
    revision: str  # the exact model version (a commit of the Hugging Face repository)
    dimension: int  # numbers per vector
    batch_size: int  # texts per batch, as notebook 04 used
    document_prefix: str = ""  # added before each document (chunk) text
    query_prefix: str = ""  # added before each question


# Revisions: the notebooks did not record which version of each model they downloaded. These
# are the versions current on 2 November 2026, which are also the ones in the local cache.
# Each was last changed on or before 1 June 2026, and the pipeline ran after 17 June 2026, so
# they are very likely the versions the research used, but that is inferred, not recorded.
# Pinning them makes the backend's results repeatable from here on.
MODELS: dict[str, EmbeddingModel] = {
    model.name: model
    for model in (
        EmbeddingModel(
            name="MiniLM",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            dimension=384,
            batch_size=32,
        ),
        EmbeddingModel(
            name="MPNet",
            model_id="sentence-transformers/all-mpnet-base-v2",
            revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            dimension=768,
            batch_size=16,
        ),
        EmbeddingModel(
            name="E5",
            model_id="intfloat/multilingual-e5-base",
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            dimension=768,
            batch_size=16,
            # E5 expects these prefixes; notebook 04 used "passage: ", notebook 06 "query: ".
            document_prefix="passage: ",
            query_prefix="query: ",
        ),
        EmbeddingModel(
            name="BGE",
            model_id="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            dimension=384,
            batch_size=32,
            # BGE's model card suggests an instruction before queries; the research used none.
        ),
    )
}


def get_model(name: str) -> EmbeddingModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model {name!r}. Choose from: {', '.join(MODELS)}"
        ) from None
