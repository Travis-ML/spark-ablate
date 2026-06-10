"""Residual-stream activation capture.

Records hidden states at the output of the embedding module and of every
decoder layer — i.e. the residual stream as it enters layer 0 and as it
leaves each layer. Capture is position-selective (offsets from the last
non-pad token) so memory stays flat in sequence length.

Keys: "emb" for the embedding output, "L0".."L{n-1}" for decoder layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from sparkablate.hooks import _first_tensor, introspect


@dataclass
class CaptureResult:
    """activations[(layer_key, offset)] -> float32 [n_prompts, hidden] (CPU)."""

    activations: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)

    @property
    def keys(self) -> list[tuple[str, int]]:
        return list(self.activations)

    def __getitem__(self, key: tuple[str, int]) -> torch.Tensor:
        return self.activations[key]


def _last_token_indices(input_ids: torch.Tensor,
                        attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Index of the last non-pad token per row; handles left and right padding."""
    if attention_mask is None:
        return torch.full((input_ids.shape[0],), input_ids.shape[1] - 1,
                          dtype=torch.long, device=input_ids.device)
    mask = attention_mask.to(torch.long)
    return mask.shape[1] - 1 - mask.flip(1).argmax(dim=1)


class ActivationRecorder:
    """Captures residual-stream activations at selected token positions.

    ``record`` accepts batches as dicts with ``input_ids`` (and optionally
    ``attention_mask``) — e.g. tokenizer output — or plain input_ids tensors.
    """

    def __init__(self, model: nn.Module, strict: bool = True):
        self.model = model
        self.arch = introspect(model, strict=strict)
        self.embedding = model.get_input_embeddings()

    def _sites(self):
        yield "emb", self.embedding
        for i, lh in enumerate(self.arch.layers):
            yield f"L{i}", lh.layer

    @torch.no_grad()
    def record(self, batches, positions: int | tuple[int, ...] = (-1,)) -> CaptureResult:
        """positions are offsets from the last non-pad token (-1 = last)."""
        if isinstance(positions, int):
            positions = (positions,)
        if any(p >= 0 for p in positions):
            raise ValueError("positions are negative offsets from the end, e.g. (-1, -2)")

        chunks: dict[tuple[str, int], list[torch.Tensor]] = {}
        last_idx: dict[str, torch.Tensor] = {}
        handles = []

        def capture(key):
            def hook(module, args, output):
                x = _first_tensor(output)
                idx = last_idx["current"]
                for off in positions:
                    pos = idx + 1 + off
                    if (pos < 0).any():
                        raise ValueError(f"offset {off} reaches before the first token")
                    sel = x[torch.arange(x.shape[0], device=x.device), pos]
                    chunks.setdefault((key, off), []).append(
                        sel.detach().to(torch.float32).cpu()
                    )
            return hook

        for key, module in self._sites():
            handles.append(module.register_forward_hook(capture(key)))

        device = next(self.model.parameters()).device
        try:
            for batch in batches:
                if isinstance(batch, torch.Tensor):
                    input_ids, attention_mask = batch, None
                else:
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask")
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                last_idx["current"] = _last_token_indices(input_ids, attention_mask)
                kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
                self.model(input_ids=input_ids, use_cache=False, **kwargs)
        finally:
            for h in handles:
                h.remove()

        return CaptureResult({k: torch.cat(v, dim=0) for k, v in chunks.items()})
