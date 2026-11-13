# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/layers.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Shared transformer layers of the video backbone.

The encoder tower and the decoder tower use the same layers: a zero-centred
RMSNorm, rotary position embedding, grouped-query attention with a per-head
output gate, a SwiGLU feed-forward or a mixture-of-experts feed-forward, and a
block stack that threads the rotary table, the attention mask and the optional
per-token modulation.

The inference math follows the original implementation op by op, so bf16
outputs are bitwise equal. The comments mark every float32 and float64 island.

Two custom Triton kernels (``praxis.kernels.layers_triton``) compute the
expert SwiGLU and the decoder modulation. The original implementation runs
the same kernels.

Training uses the same forward code. Five ops get a custom backward. Thus the
gradients follow the original training code. The ops are the two Triton
kernels, the float32 router GEMM (``_RouterLogits``), the row permutation
of the expert outputs (``_PermuteRows``) and the chunked expert dispatch
(``_ChunkedRoutedExperts``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from lerobot.policies.praxis.praxis_core.kernels.layers_triton import (
    HAS_TRITON,
    modulate_fused,
    swiglu_packed,
    swiglu_packed_backward_fused,
)

ModulationDtype = Literal["bfloat16", "float32", "mixed"]
AttentionBackend = Literal["triton", "flash"]

# One compiled flex-attention callable for the whole process. The eager
# flex_attention falls back to a dense kernel that does not fit long sequences.
_compiled_flex_attention = torch.compile(flex_attention)
# Shape-polymorphic twin for callers whose sequence lengths change from call to
# call (the action expert in the "flex_dynamic" mode). It has its own compile
# cache, so its entries do not count against the static one.
#
# Short queries go to the flex decoding kernel. Its split combine is a
# persistent reduction, and by default inductor benchmarks several block
# configs for it at run time. The chosen config changes the summation order,
# so two processes can give outputs that differ in the last bits. The option
# below makes inductor pick one fixed config by a rule (no benchmark). The
# original implementation runs with the same option.
_compiled_flex_attention_dynamic = torch.compile(
    flex_attention,
    dynamic=True,
    options={"test_configs.force_filter_reduction_configs": True},
)
_FLEX_ATTENTION_COMPILED = _compiled_flex_attention


def _dense_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    kernel_options: dict[str, object] | None = None,
) -> Tensor:
    """The attention of ``flex_attention`` through a dense mask and SDPA (LeRobot addition).

    Flex attention has no backward pass on the CPU, so CPU training (the tests) runs the
    backbone attention here. The mask is the block mask's ``mask_mod`` evaluated on every
    (query, key) pair: the same attention, in the reduction order of SDPA. Rows without a
    key (none in the chunk-causal layout) give zeros, as in flex attention.
    """
    mask = None
    if block_mask is not None:
        qi = torch.arange(q.shape[-2], device=q.device)[:, None]
        ki = torch.arange(k.shape[-2], device=q.device)[None, :]
        mask = block_mask.mask_mod(None, None, qi, ki)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=enable_gqa)
    return torch.nan_to_num(out) if mask is not None else out


KeyRanges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AttentionBlocks:
    """A mask as runs of query rows that attend the same few contiguous key ranges (LeRobot addition).

    ``runs`` holds ``(row_start, row_end, key_ranges)`` for every run of equal mask rows, in row order.
    The chunk-causal masks of the towers have a few such runs, so their attention is a few unmasked
    attentions over slices of the keys (``_split_flex_attention``).
    """

    runs: tuple[tuple[int, int, KeyRanges], ...]

    @classmethod
    def from_mask(cls, mask: Tensor) -> AttentionBlocks:
        """The runs of a dense ``(queries, keys)`` bool mask."""
        mask = mask.cpu()
        changes = (mask[1:] != mask[:-1]).any(dim=1).nonzero().flatten() + 1
        starts = [0, *changes.tolist()]
        runs = []
        for start, end in zip(starts, [*starts[1:], mask.shape[0]], strict=True):
            edges = torch.diff(F.pad(mask[start].to(torch.int8), (1, 1))).nonzero().flatten().tolist()
            runs.append((start, end, tuple(zip(edges[::2], edges[1::2], strict=True))))
        return cls(tuple(runs))

    @classmethod
    def from_block_mask(cls, block_mask: BlockMask) -> AttentionBlocks:
        """The runs of the block mask's ``mask_mod`` over all (query, key) pairs."""
        queries, keys = block_mask.seq_lengths
        device = block_mask.kv_num_blocks.device
        qi = torch.arange(queries, device=device)[:, None]
        ki = torch.arange(keys, device=device)[None, :]
        return cls.from_mask(block_mask.mask_mod(None, None, qi, ki))

    def rows(self, start: int, end: int) -> AttentionBlocks:
        """The runs of the query rows ``start:end`` alone, numbered from ``start``. The keys stay the same."""
        runs = []
        for row_start, row_end, ranges in self.runs:
            row_start, row_end = max(row_start, start), min(row_end, end)
            if row_start < row_end:
                runs.append((row_start - start, row_end - start, ranges))
        return AttentionBlocks(tuple(runs))

    def attended_from_outside(self, start: int, end: int) -> bool:
        """Whether a query row outside ``start:end`` attends a key inside it."""
        return any(
            begin < end and stop > start
            for row_start, row_end, ranges in self.runs
            if row_start < start or row_end > end
            for begin, stop in ranges
        )


class ContextCache:
    """The context rows' share of a tower's calls, kept from a call over the whole sequence (LeRobot addition).

    When no other row attends the rows ``start:end`` of a sequence and only those rows change from call to
    call, as the encoder's noised chunks over the Euler steps, the other rows (the context) have the same
    keys, values and outputs on every call. The first call runs the whole sequence, and ``whole`` keeps what
    it is given. The tower then sets ``rows``, and later calls run only the rows ``start:end``: each attention
    layer computes their keys and values and takes the context's from the cache (the original implementation's
    encoder context reuse).
    """

    def __init__(self) -> None:
        self.rows: tuple[int, int] | None = None
        self._kept: dict[tuple[object, str], Tensor] = {}

    def whole(self, owner: object, name: str, t: Tensor, dim: int) -> Tensor:
        """``t`` over the whole sequence along ``dim``: as it is before ``rows`` is set, which keeps it under
        ``(owner, name)``, and afterwards the kept tensor with its rows ``start:end`` replaced by ``t``."""
        key = (owner, name)
        if self.rows is None:
            self._kept[key] = t
            return t
        start, end = self.rows
        kept = self._kept[key]
        after = kept.narrow(dim, end, kept.size(dim) - end)
        return torch.cat([kept.narrow(dim, 0, start), t, after], dim=dim)


def _key_slices(t: Tensor, ranges: KeyRanges) -> Tensor:
    if len(ranges) == 1:
        return t[:, :, ranges[0][0] : ranges[0][1]]
    return torch.cat([t[:, :, start:end] for start, end in ranges], dim=2)


def _split_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_mask: AttentionBlocks | BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    kernel_options: dict[str, object] | None = None,
) -> Tensor:
    """The attention of ``_dense_flex_attention`` as unmasked SDPAs over the mask's runs (LeRobot addition).

    A dense masked SDPA computes every (query, key) pair, also the masked ones: about twice the work
    on the towers' masks. Here each run of rows attends only its key ranges, and runs of the same
    shape go through one SDPA as a batch. Masked keys add nothing to the softmax, so this is the same
    attention in another reduction order. A ``BlockMask`` falls back to the dense form.
    """
    if not isinstance(block_mask, AttentionBlocks):
        return _dense_flex_attention(q, k, v, block_mask, scale, enable_gqa, kernel_options)
    groups: dict[tuple[int, int], list[tuple[int, KeyRanges]]] = {}
    for start, end, ranges in block_mask.runs:
        size = sum(stop - begin for begin, stop in ranges)
        groups.setdefault((end - start, size), []).append((start, ranges))
    batch = q.shape[0]
    pieces: dict[int, Tensor] = {}
    for (rows, size), members in groups.items():
        if size == 0:
            # Rows without a key give zeros, as in flex attention.
            for start, _ in members:
                pieces[start] = q.new_zeros(*q.shape[:2], rows, v.shape[-1])
            continue
        out = F.scaled_dot_product_attention(
            torch.cat([q[:, :, start : start + rows] for start, _ in members]),
            torch.cat([_key_slices(k, ranges) for _, ranges in members]),
            torch.cat([_key_slices(v, ranges) for _, ranges in members]),
            scale=scale,
            enable_gqa=enable_gqa,
        )
        for (start, _), piece in zip(members, out.split(batch), strict=True):
            pieces[start] = piece
    return torch.cat([pieces[start] for start in sorted(pieces)], dim=2)


FlexAttentionMode = Literal["compiled", "eager", "dense", "split"]
_FLEX_ATTENTION_MODE: FlexAttentionMode = "compiled"


def set_flex_attention_mode(mode: FlexAttentionMode) -> None:
    """Select the flex attention of the backbone: compiled (default), eager, the dense SDPA form, or
    the split SDPA form.

    The compiled CPU kernel fails to build for some chunk-causal masks, and the eager kernel has
    no CPU backward; CPU runs (tests, CPU-only hosts) select ``"dense"``. ``"split"`` runs the dense
    form without a mask, a few SDPAs over the runs of the mask (``_split_flex_attention``), for
    compilers that see only PyTorch operations. The towers then pass ``AttentionBlocks`` for their
    masks. All four compute the same attention. ``Attention.forward`` reads the module attribute at
    call time.
    """
    global _compiled_flex_attention, _FLEX_ATTENTION_MODE
    options = {
        "compiled": _FLEX_ATTENTION_COMPILED,
        "eager": flex_attention,
        "dense": _dense_flex_attention,
        "split": _split_flex_attention,
    }
    _compiled_flex_attention = options[mode]
    _FLEX_ATTENTION_MODE = mode


def flex_attention_mode() -> FlexAttentionMode:
    """The flex attention of the backbone (see ``set_flex_attention_mode``)."""
    return _FLEX_ATTENTION_MODE


def set_flex_attention_compiled(compiled: bool) -> None:
    """Select the compiled (default) or the dense CPU flex attention (see ``set_flex_attention_mode``)."""
    set_flex_attention_mode("compiled" if compiled else "dense")


_CUSTOM_KERNELS = True


def set_custom_kernels(enabled: bool) -> None:
    """Run the custom Triton kernels on CUDA (default) or the PyTorch ops they stand for (LeRobot addition).

    The kernels are those of ``modulate`` and of the expert SwiGLU. ``torch.export`` records only PyTorch ops,
    so the export of the policy runs without them. The PyTorch ops compute the same values with other rounding.
    """
    global _CUSTOM_KERNELS
    _CUSTOM_KERNELS = enabled


def custom_kernels_enabled() -> bool:
    """Whether the custom Triton kernels run on CUDA (see ``set_custom_kernels``)."""
    return _CUSTOM_KERNELS


# The router GEMM runs over row chunks of this size. A different chunking can
# make cuBLAS pick a different kernel, which can flip near-tied experts.
_ROUTER_CHUNK_ROWS = 65536


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------

_MODULATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _modulate_eager(
    x: Tensor,
    shift: Tensor,
    scale: Tensor,
    compute: Literal["bfloat16", "float32"] | None,
) -> Tensor:
    orig_dtype = x.dtype
    if compute is not None:
        dt = torch.float32 if compute == "float32" else torch.bfloat16
        x, shift, scale = x.to(dt), shift.to(dt), scale.to(dt)
    out = x * scale + shift
    if compute is not None:
        out = out.to(orig_dtype)
    return out


def _fused_modulate_case(x: Tensor, shift: Tensor, scale: Tensor) -> str | None:
    """Return "same", "rows" or None for the fused modulate kernel."""
    tensors = (x, shift, scale)
    if not all(t.is_cuda and t.device == x.device for t in tensors):
        return None
    if not all(t.dtype in _MODULATE_DTYPES and t.ndim == 3 for t in tensors):
        return None
    if x.shape == shift.shape == scale.shape:
        return "same"
    if (
        shift.shape == scale.shape
        and shift.shape[0] == x.shape[0]
        and shift.shape[1] == 1
        and shift.shape[2] == x.shape[2]
    ):
        return "rows"
    return None


def modulate(
    x: Tensor,
    shift: Tensor,
    scale: Tensor,
    dtype: ModulationDtype | None = None,
) -> Tensor:
    """Return ``x * scale + shift`` in the dtype of ``x``.

    ``"mixed"`` and ``"float32"`` compute in float32. ``"bfloat16"`` computes
    in bf16. ``None`` uses the normal type promotion. With ``"mixed"``, 3-D
    CUDA inputs of one shape (or ``shift`` and ``scale`` of shape
    ``(B, 1, C)``) go to a Triton kernel, unless ``set_custom_kernels`` turned
    it off. That kernel can fuse the multiply and the add, so its rounding is
    not the rounding of the eager path.
    """
    compute: Literal["bfloat16", "float32"] | None = "float32" if dtype == "mixed" else dtype
    fused = dtype == "mixed" and HAS_TRITON and _CUSTOM_KERNELS
    case = _fused_modulate_case(x, shift, scale) if fused else None
    if case is None:
        return _modulate_eager(x, shift, scale, compute)
    return modulate_fused(x, shift, scale, rows=case == "rows")


# ---------------------------------------------------------------------------
# Normalization and rotary embedding
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Zero-centred RMSNorm. The scale is ``1 + weight``.

    With ``cast_dtype`` set, the input and the weight are cast to that dtype,
    the norm runs in that dtype, and the output is cast back.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        cast_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.cast_dtype = cast_dtype
        self.weight = nn.Parameter(torch.zeros(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        orig_dtype = x.dtype
        if self.cast_dtype is not None:
            x = x.to(self.cast_dtype)
            if weight is not None:
                weight = weight.to(self.cast_dtype)
        if weight is not None:
            weight = 1 + weight
        out = F.rms_norm(x, self.normalized_shape, weight=weight, eps=self.eps)
        if self.cast_dtype is not None:
            out = out.to(orig_dtype)
        return out


class RotaryEmbedding(nn.Module):
    """Rotary (cos, sin) table for integer positions.

    Only the first ``effective_dim // 2`` frequency pairs rotate. The other
    pairs have zero frequency and pass through unchanged.

    A position is split as ``high * table_size + low``. For each part the
    angle is a float64 product, and cos and sin are taken in float64 on the
    device and rounded to float32. The two parts are then combined with the
    angle-sum identity in float64 and rounded to float32. This equals a lookup
    in precomputed ``table_size``-row tables, bit for bit.
    """

    def __init__(
        self,
        dim: int = 128,
        effective_dim: int = 64,
        base: int = 26_000_000,
        table_size: int = 1_000_000,
    ) -> None:
        super().__init__()
        assert effective_dim <= dim
        self.dim = dim
        self.effective_dim = effective_dim
        self.base = base
        self.table_size = table_size
        # The frequencies are computed on the CPU in float32. This plain
        # attribute is not a buffer, so module casts and moves do not touch it.
        exponent = (
            torch.arange(0, effective_dim, 2, device="cpu")[: effective_dim // 2].float() / effective_dim
        )
        theta = 1.0 / (base**exponent)
        pad = torch.zeros((dim - effective_dim) // 2, device="cpu")
        self.theta = torch.cat([theta, pad], dim=0)

    @staticmethod
    def _cos_sin(angle: Tensor) -> tuple[Tensor, Tensor]:
        """Return float64 cos and sin, each rounded to float32 and back."""
        return angle.cos().float().double(), angle.sin().float().double()

    def forward(self, positions: Tensor) -> Tensor:
        """Map ``positions`` (B, N) to a float32 table (B, N, dim // 2, 2)."""
        theta = self.theta.to(positions.device).double()
        high = positions // self.table_size
        low = positions % self.table_size
        # Table rows hold float32 positions. Both casts below are exact.
        high_pos = (high.to(torch.float32) * self.table_size).double()
        low_pos = low.to(torch.float32).double()
        cos_high, sin_high = self._cos_sin(high_pos[..., None] * theta)
        cos_low, sin_low = self._cos_sin(low_pos[..., None] * theta)
        cos = cos_high * cos_low - sin_high * sin_low
        sin = sin_high * cos_low + cos_high * sin_low
        table = torch.stack([cos, sin], dim=-1).float()
        if table.dim() == 3:
            table = table.unsqueeze(0)
        return table


def apply_rope(q: Tensor, k: Tensor, table: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate ``q`` and ``k`` (B, H, N, D) in float32.

    Channel ``j`` pairs with channel ``j + D // 2``. The outputs have the
    input dtype.
    """
    orig_dtype = q.dtype
    q0, q1 = q.float().chunk(2, dim=-1)
    k0, k1 = k.float().chunk(2, dim=-1)
    seq_len, half = q0.shape[2], q0.shape[3]
    cos, sin = table.float().chunk(2, dim=-1)
    assert cos.shape[1] == seq_len and cos.shape[2] == half and cos.shape[3] == 1
    cos = cos.reshape(-1, 1, seq_len, half)
    sin = sin.reshape(-1, 1, seq_len, half)
    q_out = torch.stack([q0 * cos - q1 * sin, q1 * cos + q0 * sin], dim=-2)
    k_out = torch.stack([k0 * cos - k1 * sin, k1 * cos + k0 * sin], dim=-2)
    return q_out.flatten(-2).to(orig_dtype), k_out.flatten(-2).to(orig_dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class QKVProjection(nn.Module):
    """Separate bias-free projections for queries, keys and values."""

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.q_weight = nn.Parameter(torch.empty(num_heads * head_dim, dim))
        self.k_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))
        self.v_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = F.linear(x, self.q_weight)
        k = F.linear(x, self.k_weight)
        v = F.linear(x, self.v_weight)
        return q, k, v


class Attention(nn.Module):
    """Grouped-query self-attention with a sigmoid gate per head.

    The order is: project, rotate q and k, RMS-normalize q and k per head,
    run flex attention, multiply by the gate, project out.
    """

    def __init__(
        self,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        backend: AttentionBackend = "triton",
    ):
        super().__init__()
        self.backend: AttentionBackend
        self._kernel_options: dict[str, object] | None
        self.set_backend(backend)
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.num_kv_heads = num_kv_heads
        self.in_proj = QKVProjection(dim, head_dim, self.num_heads, num_kv_heads)
        self.gate_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out_proj = nn.Linear(self.num_heads * head_dim, dim)

    def set_backend(self, backend: AttentionBackend) -> None:
        """Select the kernel of the flex attention: ``"flash"`` runs the CuTe flash kernel (it needs
        the optional flash-attention package and full blocks in the block mask), ``"triton"`` the
        Triton kernel. ``forward`` reads the choice at call time."""
        if backend not in ("triton", "flash"):
            raise ValueError(f"unknown attention backend {backend!r}")
        self.backend = backend
        self._kernel_options = {"BACKEND": "FLASH"} if backend == "flash" else None

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None,
        block_mask: BlockMask | AttentionBlocks | None,
        context: ContextCache | None = None,
    ) -> Tensor:
        """With ``context``, the keys and values of the whole sequence come through ``ContextCache.whole``."""
        b, s, _ = x.shape
        d = self.head_dim
        q, k, v = self.in_proj(x)
        g = self.gate_proj(x).sigmoid()
        g = g.permute(0, 2, 1).reshape(b, self.num_heads, s, 1)
        q = q.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        k = k.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        v = v.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if context is not None:
            k, v = context.whole(self, "k", k, 2), context.whole(self, "v", v, 2)
        out = _compiled_flex_attention(  # ty: ignore[no-matching-overload]
            q,
            k,
            v,
            block_mask=block_mask,
            scale=None,
            enable_gqa=k.shape[1] != q.shape[1],
            kernel_options=self._kernel_options,
        )
        out = out * g
        out = out.permute(0, 2, 1, 3).reshape(b, s, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward layers
# ---------------------------------------------------------------------------


class SwiGLUProjection(nn.Module):
    """The two bias-free input projections of a SwiGLU."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.a_weight = nn.Parameter(torch.empty(hidden, dim))
        self.b_weight = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return F.linear(x, self.a_weight), F.linear(x, self.b_weight)


class SwiGLU(nn.Module):
    """``silu(x @ a.T) * (x @ b.T)``."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.linear = SwiGLUProjection(dim, hidden)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.linear(x)
        return F.silu(a) * b


class SwiGLUFeedForward(nn.Sequential):
    """Dense SwiGLU feed-forward: ``[0]`` is the SwiGLU, ``[1]`` projects out."""

    def __init__(self, dim: int = 3072, hidden: int = 11520) -> None:
        super().__init__(SwiGLU(dim, hidden), nn.Linear(hidden, dim, bias=False))


class RoutedExperts(nn.Module):
    """Expert weight banks, expert-first: gate|up (E, C, 2H), down (E, H, C)."""

    def __init__(self, num_experts: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_weight = nn.Parameter(torch.empty(num_experts, dim, 2 * hidden))
        self.down_weight = nn.Parameter(torch.empty(num_experts, hidden, dim))


class SharedExpert(nn.Module):
    """The always-on expert, a dense SwiGLU with packed gate|up weights."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Parameter(torch.empty(dim, 2 * hidden))
        self.down = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        gate_up = x @ self.gate_up
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down


def _expert_swiglu(gate_up: Tensor) -> Tensor:
    """``silu(gate) * up`` of the routed experts.

    On CUDA this runs the custom kernel of the original implementation, unless
    ``set_custom_kernels`` turned it off. Its sigmoid rounds differently from ``F.silu``.
    """
    if gate_up.is_cuda and HAS_TRITON and _CUSTOM_KERNELS:
        return swiglu_packed(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _router_logits_forward(weight: Tensor, x: Tensor) -> Tensor:
    w = weight.to(torch.float32)
    out = x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.float32)
    for start in range(0, x.shape[0], _ROUTER_CHUNK_ROWS):
        end = min(start + _ROUTER_CHUNK_ROWS, x.shape[0])
        out[start:end] = F.linear(x[start:end].to(torch.float32), w)
    return out


class _RouterLogits(torch.autograd.Function):
    """The router GEMM with the backward of the original training code.

    The backward runs two float32 GEMMs per row chunk:
    ``dx = (g @ w).to(x.dtype)`` and ``dw += g.T @ x``. ``dw`` starts at zero
    in float32 and rounds once to the dtype of the weight.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(x, weight)
        return _router_logits_forward(weight, x)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor, Tensor]:
        x, weight = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        w = weight.to(torch.float32)
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, x.shape[0], _ROUTER_CHUNK_ROWS):
            end = min(start + _ROUTER_CHUNK_ROWS, x.shape[0])
            g = grad[start:end].to(torch.float32)
            grad_x[start:end] = (g @ w).to(x.dtype)
            grad_w += g.t() @ x[start:end].to(torch.float32)
        return grad_x, grad_w.to(weight.dtype)


def _router_logits(weight: Tensor, x: Tensor) -> Tensor:
    """Float32 router logits, computed over fixed row chunks."""
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _RouterLogits.apply(x, weight)
    return _router_logits_forward(weight, x)


class _PermuteRows(torch.autograd.Function):
    """``src[index]`` for a row permutation ``index``.

    The backward is the gather ``grad[inverse]`` with the inverse permutation,
    as in the original training code. A scatter-add backward gives the same values, but it
    turns a gradient of -0.0 into +0.0.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        src: Tensor,
        index: Tensor,
        inverse: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(inverse)
        return src[index]

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor, None, None]:
        (inverse,) = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        return grad[inverse], None, None


def _dispatch_rows(top_idx: Tensor, gate: Tensor, num_experts: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sort the (token, expert) rows of one chunk by expert.

    Returns the sort order, the token and the gate of each sorted row and the
    int32 end offsets of the expert groups.
    """
    num_tokens, top_k = top_idx.shape
    device = top_idx.device
    experts = top_idx.reshape(-1)
    _, order = experts.sort(stable=True)
    tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    counts = torch.zeros(num_experts, device=device, dtype=torch.int64)
    counts.scatter_add_(0, experts.long(), torch.ones_like(experts, dtype=torch.int64))
    offsets = counts.cumsum(dim=0).to(torch.int32)
    return order, tokens[order], gate.reshape(-1)[order], offsets


def _sum_rows_of_tokens(rows: Tensor, rows_of_token: Tensor, scale: Tensor) -> Tensor:
    """``out[t] = sum_k rows[r] * scale[t, k]`` with ``r = rows_of_token[t, k]``.

    The sum runs in float32 with four interleaved accumulators: slot ``s``
    goes to accumulator ``s % 4``, then ``((a0 + a1) + a2) + a3``. This is the
    reduction of the fused combine kernel of the original implementation.
    """
    num_tokens, top_k = rows_of_token.shape
    acc = [torch.zeros(num_tokens, rows.size(1), device=rows.device, dtype=torch.float32) for _ in range(4)]
    for slot in range(top_k):
        r = rows_of_token[:, slot]
        acc[slot % 4] = acc[slot % 4] + rows[r].float() * scale[:, slot].unsqueeze(-1)
    return ((acc[0] + acc[1]) + acc[2]) + acc[3]


TopKSum = Callable[[Tensor, Tensor, Tensor], Tensor]
_TOP_K_SUM: TopKSum | None = None


def set_top_k_sum(fn: TopKSum | None) -> None:
    """Run the forward top-k sum of the chunked expert dispatch with ``fn`` (LeRobot addition).

    ``fn`` takes the arguments of ``_sum_rows_of_tokens`` and returns its values. An export uses it to give the
    sum to its compiler as one operation. None restores ``_sum_rows_of_tokens``. ``_routed_chunk_group`` reads the
    module attribute at call time.
    """
    global _TOP_K_SUM
    _TOP_K_SUM = fn


def _routed_chunk_group(
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    sizes: Sequence[int],
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
) -> list[Tensor]:
    """Float32 scaled routed sum of each token chunk of ``x`` (see ``MoEFeedForward``). The chunks follow each
    other in ``x``, with ``sizes`` tokens each.

    The rows of all chunks are sorted by expert together (inside an expert, the rows of the first chunk
    first), so the two grouped GEMMs read each expert's weights once for all chunks. Each output row of a
    grouped GEMM depends only on its own input row, and each chunk sums only its own rows, so each sum has the
    bits of its chunk run alone.
    """
    top_k = top_idx.shape[1]
    order, sorted_tokens, _, offsets = _dispatch_rows(top_idx, gate, gate_up_w.shape[0])
    rows = x[sorted_tokens].to(gate_up_w.dtype)
    gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
    out = torch._grouped_mm(_expert_swiglu(gate_up), down_w, offsets)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    top_k_sum = _TOP_K_SUM or _sum_rows_of_tokens
    sums = []
    rows_of_tokens = inverse.split([n * top_k for n in sizes])
    for rows_of_token, chunk_gate in zip(rows_of_tokens, gate.split(list(sizes)), strict=True):
        scale = chunk_gate.to(torch.float32) * scaling
        sums.append(top_k_sum(out, rows_of_token.view(chunk_gate.shape), scale))
    return sums


def _routed_chunk_backward(
    grad_out: Tensor,
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The gradients of ``_routed_chunk``, as in the original training code.

    The chunk recomputes ``gate_up``. Returns the float32 gradients of ``x``
    and of the gates and the gradients of the two expert weight banks (in
    the weight dtype).
    """
    num_tokens, top_k = top_idx.shape
    order, sorted_tokens, sorted_gate, offsets = _dispatch_rows(top_idx, gate, gate_up_w.shape[0])
    rows = x[sorted_tokens].to(gate_up_w.dtype)
    gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
    scale = sorted_gate.to(torch.float32) * scaling
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    grad_rows_out = grad_out[sorted_tokens]
    hidden = _expert_swiglu(gate_up)
    grad_hidden_unscaled = torch._grouped_mm(grad_rows_out.to(gate_up.dtype), down_w.transpose(1, 2), offsets)
    grad_scale = (grad_hidden_unscaled * hidden).sum(dim=-1, dtype=torch.float32)
    del grad_hidden_unscaled
    grad_rows_scaled = (grad_rows_out.float() * scale.unsqueeze(-1)).to(gate_up.dtype)
    del grad_rows_out
    grad_down = torch._grouped_mm(hidden.t(), grad_rows_scaled, offsets)
    del hidden
    grad_hidden = torch._grouped_mm(grad_rows_scaled, down_w.transpose(1, 2), offsets)
    del grad_rows_scaled
    grad_gate_up = swiglu_packed_backward_fused(gate_up, grad_hidden)
    del grad_hidden
    grad_gate_up_w = torch._grouped_mm(rows.t(), grad_gate_up, offsets)
    grad_rows_in = torch._grouped_mm(grad_gate_up, gate_up_w.transpose(1, 2), offsets)
    del grad_gate_up
    ones = torch.ones(num_tokens, top_k, device=x.device, dtype=torch.float32)
    grad_x = _sum_rows_of_tokens(grad_rows_in, inverse.view(num_tokens, top_k), ones)
    routed = grad_scale * scaling if scaling != 1.0 else grad_scale
    grad_flat = torch.empty_like(routed)
    grad_flat[order] = routed
    return grad_x, grad_flat.view(num_tokens, top_k), grad_gate_up_w, grad_down


class _ChunkedRoutedExperts(torch.autograd.Function):
    """The chunked expert dispatch with the backward of the original training code.

    The forward is ``MoEFeedForward._routed_forward_chunked``. It saves only
    its inputs. The backward runs the same token chunks again: it recomputes
    the first grouped GEMM, takes the gate gradients from the SwiGLU outputs
    and sums the weight gradients of the chunks in the weight dtype.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        gate_up_w: Tensor,
        down_w: Tensor,
        scaling: float,
        chunk: int,
    ) -> Tensor:
        ctx.save_for_backward(x, top_idx, gate, gate_up_w, down_w)
        ctx.scaling, ctx.chunk = scaling, chunk  # ty: ignore[unresolved-attribute]
        return _routed_chunks(x, top_idx, gate, gate_up_w, down_w, scaling, chunk)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad_out: Tensor
    ) -> tuple[Tensor, None, Tensor, Tensor, Tensor, None, None]:
        x, top_idx, gate, gate_up_w, down_w = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        scaling, step = ctx.scaling, ctx.chunk  # ty: ignore[unresolved-attribute]
        grad_out = grad_out.contiguous()
        num_tokens = x.shape[0]
        grad_x = torch.empty_like(x)
        grad_gate = torch.empty_like(gate, dtype=torch.float32)
        grad_gu: Tensor | None = None
        grad_down: Tensor | None = None
        for start in range(0, max(num_tokens, 1), step):
            end = min(start + step, num_tokens)
            if end <= start:
                break
            gx, gg, ggu, gd = _routed_chunk_backward(
                grad_out[start:end],
                x[start:end],
                top_idx[start:end],
                gate[start:end],
                gate_up_w,
                down_w,
                scaling,
            )
            grad_x[start:end] = gx.to(x.dtype)
            grad_gate[start:end] = gg
            # The first chunk gives the sum. Later chunks add in the weight
            # dtype, as in the original training code.
            grad_gu = ggu.to(gate_up_w.dtype) if grad_gu is None else grad_gu.add_(ggu)
            grad_down = gd.to(down_w.dtype) if grad_down is None else grad_down.add_(gd)
        if grad_gu is None or grad_down is None:
            grad_gu, grad_down = torch.zeros_like(gate_up_w), torch.zeros_like(down_w)
        return (
            grad_x,
            None,
            grad_gate.to(gate.dtype),
            grad_gu.to(gate_up_w.dtype),
            grad_down.to(down_w.dtype),
            None,
            None,
        )


def _rows_of_spans(t: Tensor, spans: Sequence[tuple[int, int]]) -> Tensor:
    """The rows ``t[a : a + n]`` of each ``(a, n)`` of ``spans``, in order: one slice if they are adjacent."""
    if all(a + n == b for (a, n), (b, _) in zip(spans, spans[1:], strict=False)):
        return t.narrow(0, spans[0][0], sum(n for _, n in spans))
    return torch.cat([t.narrow(0, a, n) for a, n in spans])


def _routed_chunks(
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
    step: int,
) -> Tensor:
    """The scaled routed sum over token chunks of ``step``, rounded once."""
    return _routed_chunks_group(x, top_idx, gate, [x.shape[0]], gate_up_w, down_w, scaling, step)[0]


def _routed_chunks_group(
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    lengths: Sequence[int],
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
    step: int,
) -> list[Tensor]:
    """``_routed_chunks`` of each input. The inputs follow each other in ``x``, with ``lengths`` tokens each.

    Each input keeps its own token chunks of ``step``, and chunk ``k`` of every input runs in one
    ``_routed_chunk_group``. When those chunks are adjacent in ``x``, as when every input fits in one chunk,
    they are one slice: Torch-TensorRT converts the expert sort only when it can trace its keys back to
    ``torch.topk`` through views, which a concatenation hides.
    """
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    outs = [x.new_empty(n, x.shape[1], dtype=torch.float32) for n in lengths]
    for start in range(0, max([*lengths, 1]), step):
        group = [i for i, n in enumerate(lengths) if n > start]
        if not group:
            break
        spans = [(starts[i] + start, min(step, lengths[i] - start)) for i in group]
        sums = _routed_chunk_group(
            _rows_of_spans(x, spans),
            _rows_of_spans(top_idx, spans),
            _rows_of_spans(gate, spans),
            [n for _, n in spans],
            gate_up_w,
            down_w,
            scaling,
        )
        for i, (_, n), chunk_sum in zip(group, spans, sums, strict=True):
            outs[i][start : start + n] = chunk_sum
    return [out.to(x.dtype) for out in outs]


RoutingHook = Callable[["MoEFeedForward", Tensor, Tensor], None]


class MoEFeedForward(nn.Module):
    """Mixture of experts: one shared expert plus ``top_k`` of ``num_experts``.

    The router runs in float32 with sigmoid scores. The routing bias changes
    which experts are picked but not their weights. The picked scores are
    normalized to sum to one and cast to the activation dtype. Each expert is
    a SwiGLU. The routed sum is scaled by ``routed_scaling_factor`` and added
    to the shared expert output.

    ``expert_bias`` stays float32 under module dtype casts.

    ``routing_hook`` is None at inference. Training code can set it to a
    callable ``hook(layer, logits, top_idx)``. The layer calls it after the
    routing and before the experts, with the float32 router logits and the
    picked experts.
    """

    expert_bias: Tensor
    routing_hook: RoutingHook | None = None

    def __init__(
        self,
        dim: int = 3072,
        hidden: int = 1280,
        num_experts: int = 128,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        dispatch_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if dispatch_chunk_size is not None and dispatch_chunk_size <= 0:
            raise ValueError("dispatch_chunk_size must be positive")
        self.dispatch_chunk_size = dispatch_chunk_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.routed = RoutedExperts(num_experts, dim, hidden)
        self.shared = SharedExpert(dim, hidden)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        bias = self.expert_bias
        super()._apply(fn, recurse)
        # Keep the routing bias in float32. Move it only to the new device.
        self.expert_bias = bias.to(device=self.expert_bias.device)
        return self

    def _route(self, logits: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """The picked experts of each token and their gates in ``dtype``, from the float32 router logits."""
        scores = torch.sigmoid(logits)
        biased = scores + self.expert_bias.to(scores.dtype)
        _, top_idx = torch.topk(biased, self.top_k, dim=-1)
        picked = scores.gather(dim=-1, index=top_idx)
        gate = picked / picked.sum(dim=-1, keepdim=True)
        return top_idx, gate.to(dtype)

    def forward_pair(self, xs: Sequence[Tensor]) -> list[Tensor]:
        """``[self(x) for x in xs]`` with one pass of the routed experts for all inputs.

        The router GEMM and the shared expert run per input, as in ``forward``, and the rest of the routing
        runs row by row. Token chunk ``k`` of every input is dispatched together (``_routed_chunks_group``),
        so the grouped GEMMs read each expert's weights once for all inputs, and every output has the bits of
        ``forward``. Without ``dispatch_chunk_size`` or with grad, the inputs run one at a time. An input longer
        than one token chunk makes each chunk's sort keys a concatenation of top-k slices, which Torch-TensorRT
        may leave outside the engine; the outputs stay the same, and the released DROID checkpoint's chunk of
        16384 tokens holds a whole caption (about 3.5k tokens).
        """
        if self.dispatch_chunk_size is None or torch.is_grad_enabled():
            return [self(x) for x in xs]
        flats = [x.reshape(-1, x.size(-1)) for x in xs]
        lengths = [x_flat.shape[0] for x_flat in flats]
        logits = torch.cat([_router_logits(self.router.weight, x_flat) for x_flat in flats])
        top_idx, gate = self._route(logits, flats[0].dtype)
        if self.routing_hook is not None:
            for rows in zip(logits.split(lengths), top_idx.split(lengths), strict=True):
                self.routing_hook(self, *rows)
        shared = [self.shared(x_flat) for x_flat in flats]
        routed = _routed_chunks_group(
            torch.cat(flats),
            top_idx,
            gate,
            lengths,
            self.routed.gate_up_weight,
            self.routed.down_weight,
            float(self.routed_scaling_factor),
            int(self.dispatch_chunk_size),
        )
        return [
            (y_routed + y_shared).reshape(x.shape)
            for y_routed, y_shared, x in zip(routed, shared, xs, strict=True)
        ]

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        logits = _router_logits(self.router.weight, x_flat)
        top_idx, gate = self._route(logits, x_flat.dtype)
        if self.routing_hook is not None:
            self.routing_hook(self, logits, top_idx)
        y_shared = self.shared(x_flat)
        if self.dispatch_chunk_size is not None:
            # The chunked path applies the scaling factor itself.
            y_routed = self._routed_forward_chunked(x_flat, top_idx, gate)
        else:
            y_routed = self._routed_forward(x_flat, top_idx, gate)
            y_routed = y_routed * self.routed_scaling_factor
        out = y_routed + y_shared
        return out.reshape(shape)

    def _routed_forward_chunked(self, x: Tensor, top_idx: Tensor, gate: Tensor) -> Tensor:
        """Routed experts over token chunks of ``dispatch_chunk_size``.

        Each chunk sorts its (token, expert) rows by expert and runs the two
        grouped GEMMs. The row scale is the gate in float32 times
        ``routed_scaling_factor``. The ``top_k`` scaled rows of a token are
        summed in float32 with four interleaved accumulators (slot ``s`` goes
        to accumulator ``s % 4``, then ``((a0 + a1) + a2) + a3``). The sum is
        rounded to the dtype of ``x`` once, after all chunks. Returns the
        scaled routed sum. With grad, ``_ChunkedRoutedExperts`` gives the
        backward of the original training code.
        """
        assert self.dispatch_chunk_size is not None
        args = (
            x,
            top_idx,
            gate,
            self.routed.gate_up_weight,
            self.routed.down_weight,
            float(self.routed_scaling_factor),
            int(self.dispatch_chunk_size),
        )
        if torch.is_grad_enabled() and (
            x.requires_grad
            or gate.requires_grad
            or self.routed.gate_up_weight.requires_grad
            or self.routed.down_weight.requires_grad
        ):
            out = _ChunkedRoutedExperts.apply(*args)
            assert isinstance(out, Tensor)
            return out
        return _routed_chunks(*args)

    def _routed_forward(self, x: Tensor, top_idx: Tensor, gate: Tensor) -> Tensor:
        """Sort the (token, expert) rows by expert, run two grouped GEMMs,
        scale each row by its gate, undo the sort and sum over ``top_k``."""
        num_tokens, c = x.shape
        device = x.device
        experts = top_idx.reshape(-1)
        tokens = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        order = experts.argsort(stable=True)
        sorted_tokens = tokens[order]
        sorted_experts = experts[order]
        sorted_gate = gate.reshape(-1)[order]
        gate_up_w = self.routed.gate_up_weight
        rows = x[sorted_tokens].to(gate_up_w.dtype)
        offsets = torch.searchsorted(
            sorted_experts,
            torch.arange(1, self.num_experts + 1, device=device),
            out_int32=True,
        )
        gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
        out = torch._grouped_mm(_expert_swiglu(gate_up), self.routed.down_weight, offsets)
        out = out * sorted_gate.unsqueeze(-1)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=device)
        if torch.is_grad_enabled() and out.requires_grad:
            unsorted = _PermuteRows.apply(out, inverse, order)
        else:
            unsorted = out[inverse]
        return unsorted.reshape(num_tokens, self.top_k, c).sum(dim=1)


# ---------------------------------------------------------------------------
# Block and stack
# ---------------------------------------------------------------------------


Modulation = tuple[nn.Module, Tensor, ModulationDtype | None]


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + attn(norm(x) * s)``, then ``x + ffn(norm(x) * s)``.

    ``s`` is ``1 / sqrt(layer_index)``. It is a persistent buffer, so the
    checkpoint value (rounded to bf16) is the value in use.

    ``modulation`` is ``(modulator, x_mod, dtype)`` in a modulated stack. The
    block then first replaces ``x`` with ``modulate(x, shift, 1 + scale)``,
    with ``(shift, scale)`` from ``modulator(x_mod)``. Thus one block call
    holds all the work of a layer. An activation checkpoint of the block
    also recomputes the modulation.
    """

    s: Tensor

    def __init__(
        self,
        ffn: nn.Module,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        layer_index: int = 1,
        attention_backend: AttentionBackend = "triton",
    ) -> None:
        super().__init__()
        self.register_buffer("s", torch.tensor(1 / (layer_index**0.5)))
        self.ln_in = RMSNorm(dim)
        self.mha = Attention(dim, head_dim, num_kv_heads, attention_backend)
        self.ln_out = RMSNorm(dim)
        self.ffn = ffn

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None,
        block_mask: BlockMask | AttentionBlocks | None = None,
        modulation: Modulation | None = None,
        context: ContextCache | None = None,
    ) -> Tensor:
        x = self._attend(x, rope, block_mask, modulation, context)
        return x + self.ffn(self.ln_out(x) * self.s)

    def forward_pair(
        self,
        xs: Sequence[Tensor],
        ropes: Sequence[Tensor | None],
        block_masks: Sequence[BlockMask | AttentionBlocks | None],
        modulations: Sequence[Modulation | None],
        contexts: Sequence[ContextCache | None],
    ) -> list[Tensor]:
        """``forward`` of each input, with one pass of a mixture of experts' routed experts for all inputs
        (``MoEFeedForward.forward_pair``)."""
        xs = [self._attend(*args) for args in zip(xs, ropes, block_masks, modulations, contexts, strict=True)]
        ffn_in = [self.ln_out(x) * self.s for x in xs]
        if isinstance(self.ffn, MoEFeedForward):
            ys = self.ffn.forward_pair(ffn_in)
        else:
            ys = [self.ffn(h) for h in ffn_in]
        return [x + y for x, y in zip(xs, ys, strict=True)]

    def _attend(
        self,
        x: Tensor,
        rope: Tensor | None,
        block_mask: BlockMask | AttentionBlocks | None,
        modulation: Modulation | None,
        context: ContextCache | None,
    ) -> Tensor:
        """The modulation, if any, and the attention half of the block."""
        if modulation is not None:
            modulator, x_mod, dtype = modulation
            shift, scale = modulator(x_mod).chunk(2, dim=-1)
            x = modulate(x, shift, 1 + scale, dtype=dtype)
        return x + self.mha(self.ln_in(x) * self.s, rope, block_mask, context)


class Transformer(nn.Module):
    """A stack of blocks with a shared rotary table.

    The first ``num_dense`` blocks have a dense SwiGLU, the others a mixture of
    experts. With ``mod_dim`` set, each block has a modulator: a linear map of
    the per-token conditioning ``x_mod`` to ``(shift, scale)``. The block input
    is then ``modulate(x, shift, 1 + scale)``.
    """

    def __init__(
        self,
        depth: int = 14,
        num_dense: int = 2,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        dense_hidden: int = 11520,
        moe_hidden: int = 1280,
        num_experts: int = 128,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        mod_dim: int | None = None,
        modulation_dtype: ModulationDtype | None = "mixed",
        rope_effective_dim: int = 64,
        rope_base: int = 26_000_000,
        rope_table_size: int = 1_000_000,
        attention_backend: AttentionBackend = "triton",
        moe_dispatch_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.rope = RotaryEmbedding(head_dim, rope_effective_dim, rope_base, rope_table_size)
        blocks = []
        for i in range(depth):
            ffn: nn.Module
            if i < num_dense:
                ffn = SwiGLUFeedForward(dim, dense_hidden)
            else:
                ffn = MoEFeedForward(
                    dim,
                    moe_hidden,
                    num_experts,
                    top_k,
                    routed_scaling_factor,
                    moe_dispatch_chunk_size,
                )
            blocks.append(
                TransformerBlock(
                    ffn,
                    dim,
                    head_dim,
                    num_kv_heads,
                    layer_index=i + 1,
                    attention_backend=attention_backend,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.modulators: nn.ModuleList | None = None
        if mod_dim is not None:
            self.modulators = nn.ModuleList([nn.Linear(mod_dim, 2 * dim) for _ in range(depth)])
        self.modulation_dtype: ModulationDtype | None = modulation_dtype

    def rope_table(self, rope_idx: Tensor, seq_len: int) -> Tensor:
        """Rotary table for ``rope_idx`` (B, N), cut to ``seq_len`` tokens."""
        return self.rope(rope_idx)[:, :seq_len]

    def block_forward(
        self,
        i: int,
        x: Tensor,
        rope: Tensor,
        x_mod: Tensor | None = None,
        block_mask: BlockMask | AttentionBlocks | None = None,
        context: ContextCache | None = None,
    ) -> Tensor:
        """Run block ``i``, with its modulation first when the stack has one."""
        if self.modulators is None:
            return self.blocks[i](x, rope, block_mask, context=context)
        assert x_mod is not None, "a modulated stack needs x_mod"
        modulation = (self.modulators[i], x_mod, self.modulation_dtype)
        return self.blocks[i](x, rope, block_mask, modulation, context)

    def forward(
        self,
        x: Tensor,
        rope_idx: Tensor,
        x_mod: Tensor | None = None,
        block_mask: BlockMask | AttentionBlocks | None = None,
        tap_blocks: Sequence[int] = (),
        context: ContextCache | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Map ``x`` (B, N, C) to (B, N, C) and return the taps.

        ``rope_idx`` is (B, N) int64. The taps are the outputs of the blocks in
        ``tap_blocks``, in the order of ``tap_blocks``. ``context`` goes to every
        attention layer (``Attention.forward``).
        """
        rope = self.rope_table(rope_idx, x.size(-2))
        taps: dict[int, Tensor] = {}
        for i in range(len(self.blocks)):
            x = self.block_forward(i, x, rope, x_mod, block_mask, context)
            if i in tap_blocks:
                taps[i] = x
        return x, [taps[i] for i in tap_blocks]

    def forward_pair(
        self,
        xs: Sequence[Tensor],
        rope_idxs: Sequence[Tensor],
        x_mods: Sequence[Tensor | None],
        block_masks: Sequence[BlockMask | AttentionBlocks | None],
        tap_blocks: Sequence[int] = (),
        contexts: Sequence[ContextCache | None] | None = None,
    ) -> list[tuple[Tensor, list[Tensor]]]:
        """``forward`` of each input, block by block, so each mixture-of-experts block reads its routed
        experts' weights once for all inputs (``TransformerBlock.forward_pair``)."""
        ropes = [self.rope_table(rope_idx, x.size(-2)) for rope_idx, x in zip(rope_idxs, xs, strict=True)]
        contexts = [None] * len(xs) if contexts is None else contexts
        taps: list[dict[int, Tensor]] = [{} for _ in xs]
        for i, block in enumerate(self.blocks):
            if self.modulators is None:
                modulations: list[Modulation | None] = [None] * len(xs)
            else:
                assert all(x_mod is not None for x_mod in x_mods), "a modulated stack needs x_mod"
                modulations = [(self.modulators[i], x_mod, self.modulation_dtype) for x_mod in x_mods]
            xs = block.forward_pair(xs, ropes, block_masks, modulations, contexts)
            if i in tap_blocks:
                for tap, x in zip(taps, xs, strict=True):
                    tap[i] = x
        return [(x, [tap[i] for i in tap_blocks]) for x, tap in zip(xs, taps, strict=True)]
