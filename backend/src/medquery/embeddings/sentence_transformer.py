"""Embedder backed by the `sentence-transformers` library (MiniLM, MPNet, E5, BGE).

Heavy imports (torch, sentence-transformers) happen only when a model is loaded, so
the rest of the package and the unit tests run without them.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np

from medquery.embeddings.base import Embedder, EmbedderInfo
from medquery.embeddings.specs import EmbeddingConfigError, EmbeddingModelSpec

Device = Literal["auto", "cpu", "cuda"]

_RECORDED_LIBRARIES = ("sentence-transformers", "transformers", "torch", "tokenizers")
_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


def cached_revision(model_id: str, revision: str | None, cache_folder: str | None) -> str | None:
    """Commit hash the local Hugging Face cache resolves `revision` (default: main) to."""
    if revision and _COMMIT_HASH.match(revision):
        return revision
    if cache_folder:
        cache = Path(cache_folder)
    else:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
    ref = cache / f"models--{model_id.replace('/', '--')}" / "refs" / (revision or "main")
    try:
        commit = ref.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return commit if _COMMIT_HASH.match(commit) else None


def resolve_device(device: Device) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class SentenceTransformerEmbedder(Embedder):
    """Wraps a `SentenceTransformer` model.

    Prefixes are applied explicitly by `Embedder.prepare` (as in the research
    notebooks). Models that define a *default* prompt in their configuration are
    rejected, because the library would silently add a second prefix.

    `model` may be supplied directly (used by tests with a stand-in object); otherwise
    the model is loaded by `spec.model_id`/`spec.revision`. With
    `local_files_only=True` nothing is downloaded and a missing model raises `OSError`.
    """

    def __init__(
        self,
        spec: EmbeddingModelSpec,
        *,
        device: Device = "auto",
        local_files_only: bool = False,
        cache_folder: str | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(spec)
        self._cache_folder = cache_folder
        self._loaded_from_hub = model is None
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                spec.model_id,
                revision=spec.revision,
                device=resolve_device(device),
                local_files_only=local_files_only,
                cache_folder=cache_folder,
            )
        self._model = model
        self._check_model()

    def _check_model(self) -> None:
        actual = self._model.get_embedding_dimension()
        if actual != self.spec.dimension:
            raise EmbeddingConfigError(
                f"{self.spec.model_id}: model dimension {actual} != spec dimension "
                f"{self.spec.dimension}"
            )
        default_prompt = getattr(self._model, "default_prompt_name", None)
        if default_prompt:
            raise EmbeddingConfigError(
                f"{self.spec.model_id} defines default prompt {default_prompt!r}; "
                "prefixes must come from the spec only"
            )

    @property
    def max_seq_length(self) -> int | None:
        return getattr(self._model, "max_seq_length", None)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.spec.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _token_counts(self, texts: list[str]) -> list[int]:
        encoded = self._model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def _resolved_revision(self) -> str | None:
        """Commit hash of the loaded weights, or None if it cannot be determined."""
        try:
            first = self._model[0]
        except (TypeError, IndexError, KeyError):
            first = None
        for attr in ("auto_model", "model"):
            config = getattr(getattr(first, attr, None), "config", None)
            commit = getattr(config, "_commit_hash", None)
            if isinstance(commit, str) and _COMMIT_HASH.match(commit):
                return commit
        if self._loaded_from_hub:
            return cached_revision(self.spec.model_id, self.spec.revision, self._cache_folder)
        return None

    def info(self) -> EmbedderInfo:
        device = str(getattr(self._model, "device", "unknown"))
        device_name = None
        if device.startswith("cuda"):
            import torch

            device_name = torch.cuda.get_device_name(torch.device(device))
        versions = {}
        for package in _RECORDED_LIBRARIES:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                continue
        return EmbedderInfo(
            name=self.spec.name,
            model_id=self.spec.model_id,
            requested_revision=self.spec.revision,
            resolved_revision=self._resolved_revision(),
            dimension=self.spec.dimension,
            max_seq_length=self.max_seq_length,
            normalize=self.spec.normalize,
            query_prefix=self.spec.query_prefix,
            passage_prefix=self.spec.passage_prefix,
            backend="sentence-transformers",
            device=device,
            device_name=device_name,
            library_versions=versions,
            model_prompts=dict(getattr(self._model, "prompts", None) or {}),
        )
