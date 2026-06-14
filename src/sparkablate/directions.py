"""Difference-of-means direction extraction and direction artifacts.

Implements the candidate-extraction half of the refusal-direction methodology
(Arditi et al. 2024): for each (layer, position) capture site, the candidate
direction is the difference between mean residual-stream activations over two
contrastive prompt sets, unit-normalized.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field

import torch

from sparkablate.capture import CaptureResult

ARTIFACT_VERSION = 1


def diff_of_means(pos: CaptureResult, neg: CaptureResult) -> dict[tuple[str, int], torch.Tensor]:
    """Unit-normalized (mean_pos - mean_neg) per capture key."""
    if set(pos.activations) != set(neg.activations):
        raise ValueError("contrastive captures cover different (layer, position) keys")
    out = {}
    degenerate = []
    for key in pos.activations:
        d = pos[key].mean(dim=0) - neg[key].mean(dim=0)
        norm = d.norm()
        if norm == 0:
            degenerate.append(key)  # e.g. emb @ a shared chat-template suffix token
            continue
        out[key] = d / norm
    if degenerate:
        warnings.warn(
            f"dropping {len(degenerate)} site(s) with zero difference-of-means "
            f"{degenerate}: at the embedding this is expected when chat-templated "
            "prompts share a fixed token at the captured position (the raw token "
            "embedding is identical, since the embedding does no context mixing).",
            stacklevel=2,
        )
    if not out:
        raise ValueError(
            "every capture site had a zero difference-of-means; the harmful and "
            "harmless prompt sets appear identical at the captured position(s)."
        )
    return out


def prompts_sha256(prompts: list[str]) -> str:
    h = hashlib.sha256()
    for prompt in prompts:
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass(frozen=True)
class DirectionArtifact:
    """A reusable direction vector with provenance metadata."""

    vector: torch.Tensor          # float32 [hidden], unit norm
    model_name: str
    layer_key: str                # "emb" or "L{i}"
    position: int                 # offset from last non-pad token, e.g. -1
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        torch.save(
            {
                "version": ARTIFACT_VERSION,
                "vector": self.vector.to(torch.float32).cpu(),
                "model_name": self.model_name,
                "layer_key": self.layer_key,
                "position": self.position,
                "meta": dict(self.meta),
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "DirectionArtifact":
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if raw.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported direction artifact version {raw.get('version')!r}")
        return DirectionArtifact(
            vector=raw["vector"],
            model_name=raw["model_name"],
            layer_key=raw["layer_key"],
            position=raw["position"],
            meta=raw["meta"],
        )


def make_artifact(direction: torch.Tensor, model_name: str, key: tuple[str, int],
                  harmful_prompts: list[str], harmless_prompts: list[str],
                  extra_meta: dict | None = None) -> DirectionArtifact:
    layer_key, position = key
    meta = {
        "harmful_sha256": prompts_sha256(harmful_prompts),
        "harmless_sha256": prompts_sha256(harmless_prompts),
        "n_harmful": len(harmful_prompts),
        "n_harmless": len(harmless_prompts),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta.update(extra_meta or {})
    return DirectionArtifact(
        vector=direction.to(torch.float32).cpu(),
        model_name=model_name,
        layer_key=layer_key,
        position=position,
        meta=meta,
    )


def candidate_keys(directions: dict[tuple[str, int], torch.Tensor],
                   num_layers: int, layer_fraction: float = 0.8) -> list[tuple[str, int]]:
    """Filter candidates to layers below ``layer_fraction`` of depth.

    Directions from the last ~20% of layers tend to score well on the
    training criterion but generalize poorly (Arditi et al.), so they are
    excluded from selection by default. The embedding site is kept.
    """
    cutoff = int(num_layers * layer_fraction)
    keep = []
    for layer_key, off in directions:
        if layer_key == "emb" or int(layer_key[1:]) < cutoff:
            keep.append((layer_key, off))
    return keep
