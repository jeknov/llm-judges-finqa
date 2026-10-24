"""Backend selection by profile: HF Transformers for tiny/local, vLLM on the GPU node."""

from __future__ import annotations

from functools import lru_cache

from omegaconf import DictConfig

from fqsel.backends.base import LMBackend
from fqsel.config import cache_dir


def _model_path(cfg: DictConfig) -> tuple[str, str | None]:
    base = cfg.model.base
    if base.name == "toy":
        from fqsel.backends.toy import build_toy_model

        return str(build_toy_model(cache_dir(cfg) / "toy_model")), None
    return base.name, base.revision


def make_backend(cfg: DictConfig) -> LMBackend:
    path, revision = _model_path(cfg)
    if cfg.backend == "hf":
        from fqsel.backends.hf_backend import HFBackend

        return HFBackend(path, revision=revision, dtype=cfg.model.base.dtype)
    if cfg.backend == "vllm":
        from fqsel.backends.vllm_backend import VLLMBackend

        return VLLMBackend(path, revision, cfg.model.base.dtype, cfg.model.base.min_context)
    raise ValueError(f"unknown backend {cfg.backend!r}")


@lru_cache(maxsize=4)
def _tokenizer(path: str, revision: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, revision=revision)


def load_tokenizer(cfg: DictConfig):
    """The model.base tokenizer alone (for qa_tokens), without loading weights."""
    return _tokenizer(*_model_path(cfg))
