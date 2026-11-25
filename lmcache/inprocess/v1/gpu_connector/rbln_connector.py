# SPDX-License-Identifier: Apache-2.0
"""RBLN NPU connector for LMCache.

vLLM-RBLN keeps its KV caches as ``torch`` tensors on ``device='rbln'``, so KV
moves with torch copies only. Each layer's cache is

* 6-D ``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]``, possibly
  allocated blocks-first (:func:`as_kv_first`);
* 3-D ``[num_blocks, block_size, head_dim]`` under MLA (the latent).

A chunk is head-major: ``[2, L, H, T, D]``, or ``[1, L, 1, T, D]`` under MLA.
"""

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING

# Third Party
from typing_extensions import override
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.inprocess.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

# (block, offset in block, first chunk token, stop chunk token)
Run = tuple[int, int, int, int]


def as_kv_first(kv_cache: torch.Tensor) -> torch.Tensor:
    """View a 6-D attention cache as ``[2, NB, NH, 1, BS, HD]``.

    vLLM-RBLN allocates it blocks-first (``[NB, 2, ...]``) by default and
    K/V-first on rbln_triton_ops; the transpose is stride-only. Other caches
    are returned unchanged. A two-block cache reads as K/V-first.
    """
    if kv_cache.ndim == 6 and kv_cache.shape[1] == 2 and kv_cache.shape[0] != 2:
        return kv_cache.transpose(0, 1)
    return kv_cache


def slot_runs(slots: list[int], block_size: int) -> list[Run]:
    """Split ``slots`` into ``(block, offset, token, stop)`` runs: the longest
    stretches of chunk tokens with consecutive slots in one block."""
    runs: list[Run] = []
    token = 0
    while token < len(slots):
        block, offset = divmod(slots[token], block_size)
        stop = token + 1
        while (
            stop < len(slots)
            and offset + stop - token < block_size
            and slots[stop] == slots[token] + stop - token
        ):
            stop += 1
        runs.append((block, offset, token, stop))
        token = stop
    return runs


class RBLNConnector(GPUConnectorInterface):
    """KV connector for vLLM-RBLN device-tensor caches.

    Build it with :meth:`from_metadata`. The caches are registered with
    :meth:`register_kv_caches`, or by the first transfer from the
    ``kvcaches`` the engine passes it.

    Args:
        num_layers: Number of attention layers.
        num_kv_heads: KV heads per layer (1 under MLA).
        head_dim: Head size (the latent size under MLA).
        dtype: KV cache dtype.
        use_mla: Layers are 3-D MLA latent caches and chunks lead with 1, not 2.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        use_mla: bool = False,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim_size = num_kv_heads * head_dim
        self.dtype = dtype
        self.use_mla = use_mla
        self.kv_dim = 1 if use_mla else 2
        # Read from the caches at registration.
        self.block_size: int | None = None
        # K/V-first views of the caches, and the first one as allocated.
        self._kv_caches: list[torch.Tensor] = []
        self._layout_ref: torch.Tensor | None = None
        self._device: torch.device | None = None
        # Upload staging laid out like the caches, grown on demand and reused.
        self._staging: torch.Tensor | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: torch.device | None = None,
    ) -> "RBLNConnector":
        """Build from the engine metadata. ``use_gpu`` and ``device`` are
        unused: the connector works on the device of the registered caches."""
        num_layers, _, _, num_kv_heads, head_dim = metadata.kv_shape
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=metadata.kv_dtype,
            use_mla=metadata.use_mla,
        )

    @override
    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [self.kv_dim, self.num_layers, self.num_kv_heads, num_tokens, self.head_dim]
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: Sequence[torch.Tensor]) -> None:
        """Register vLLM's caches in layer order: 6-D (either layout, see
        :func:`as_kv_first`), or 3-D under MLA."""
        if len(kv_caches) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} KV caches, got {len(kv_caches)}"
            )
        expected_ndim = 3 if self.use_mla else 6
        kv_first = [as_kv_first(kv) for kv in kv_caches]
        for i, kv in enumerate(kv_first):
            heads = 1 if self.use_mla else kv.shape[2]
            if (
                kv.ndim != expected_ndim
                or kv.dtype != self.dtype
                or (heads, kv.shape[-1]) != (self.num_kv_heads, self.head_dim)
            ):
                layout = "[NB, BS, HS]" if self.use_mla else "[2, NB, NH, 1, BS, HS]"
                raise ValueError(
                    f"KV cache {i} is {tuple(kv_caches[i].shape)} {kv.dtype}; "
                    f"expected {layout} {self.dtype} with "
                    f"{self.num_kv_heads} heads of {self.head_dim}"
                )
        self.block_size = kv_first[0].shape[1 if self.use_mla else 4]
        self._kv_caches = kv_first
        self._layout_ref = kv_caches[0]
        self._device = kv_caches[0].device
        logger.info(
            "Registered %d KV caches on %s, allocated shape=%s, block_size=%d, "
            "dtype=%s",
            len(kv_first),
            self._device,
            tuple(kv_caches[0].shape),
            self.block_size,
            self.dtype,
        )

    def _ensure_ready(self, op: str, kvcaches: Sequence[torch.Tensor] | None) -> None:
        """Register the engine's layer-ordered ``kvcaches`` on first use: the
        vLLM adapter passes them to each transfer."""
        if self._kv_caches:
            return
        if not kvcaches:
            raise RuntimeError(
                f"{op}: KV caches not registered — call register_kv_caches first"
            )
        self.register_kv_caches(list(kvcaches))

    # ------------------------------------------------------------------
    # Copies
    # ------------------------------------------------------------------

    def _staging_view(self, shape: torch.Size) -> torch.Tensor:
        """Device upload buffer for a chunk.

        Outside MLA, ``set_device_layout_like`` binds it in the cache's layout,
        so the upload is one transfer and the scatter stays on device (0.6 ms
        vs 4.5 ms per Qwen3-1.7B scatter). An MLA latent cache is not matched:
        that pulled the whole cache through the host on every scatter.
        """
        buf = self._staging
        if buf is None or buf.numel() < shape.numel():
            buf = torch.empty(shape.numel(), dtype=self.dtype, device=self._device)
            if not self.use_mla:
                # Third Party
                from torch_rbln.memory import set_device_layout_like

                set_device_layout_like(buf, self._layout_ref)
            self._staging = buf
        return buf[: shape.numel()].view(shape)

    def _runs(self, slots: torch.Tensor) -> list[Run]:
        """:func:`slot_runs` of a chunk's slots, skipping the per-token walk
        when the chunk is whole blocks (the common case)."""
        assert self.block_size is not None
        bs = self.block_size
        slots = slots.cpu()
        if (slots < 0).any():
            raise ValueError("RBLN KV transfer got a negative slot")
        n = slots.numel()
        if n and n % bs == 0:
            rows = slots.view(-1, bs)
            first = rows[:, :1]
            if (first % bs == 0).all() and (rows == first + torch.arange(bs)).all():
                blocks = (first[:, 0] // bs).tolist()
                return [(b, 0, i * bs, (i + 1) * bs) for i, b in enumerate(blocks)]
        return slot_runs(slots.tolist(), bs)

    def _run_pairs(
        self, runs: list[Run], chunk: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """``(cache views, chunk views)`` of every run and layer."""
        cache_views: list[torch.Tensor] = []
        chunk_views: list[torch.Tensor] = []
        for block, offset, token, stop in runs:
            end = offset + stop - token
            for layer, kv in enumerate(self._kv_caches):
                if self.use_mla:
                    cache_views.append(kv[block, offset:end])
                    chunk_views.append(chunk[0, layer, 0, token:stop])
                    continue
                for half in (0, 1):
                    cache_views.append(kv[half, block, :, 0, offset:end])
                    chunk_views.append(chunk[half, layer, :, token:stop])
        return cache_views, chunk_views

    def _stack_runs(self, runs: list[Run]) -> torch.Tensor:
        """Gather the 6-D caches into one ``[2, L, H, T, D]`` device tensor:
        one stack over layers per run, then a cat over runs."""
        pieces = [
            torch.stack(
                [
                    kv[:, block, :, 0, offset : offset + stop - token]
                    for kv in self._kv_caches
                ],
                dim=1,
            )
            for block, offset, token, stop in runs
        ]
        return torch.cat(pieces, dim=3) if len(pieces) > 1 else pieces[0]

    def _chunk(self, memory_obj: MemoryObj, num_tokens: int) -> torch.Tensor:
        """``memory_obj``'s tensor viewed as a head-major chunk."""
        if memory_obj.tensor is None:
            raise ValueError("RBLN KV transfer needs a memory object with a tensor")
        return memory_obj.tensor.reshape(self.get_shape(num_tokens))

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    @override
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        """Copy the KV of tokens ``[start, end)`` into ``memory_obj`` (host or
        device). Requires ``slot_mapping`` in ``kwargs``."""
        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            raise ValueError("slot_mapping is required for RBLN D2H")
        self._ensure_ready("D2H", kwargs.get("kvcaches"))
        dst = self._chunk(memory_obj, end - start)
        runs = self._runs(slot_mapping[start:end])
        if not self.use_mla and any(
            stop - token != self.block_size for _, _, token, stop in runs
        ):
            # Partial blocks: stack on device, then one transfer.
            dst.copy_(self._stack_runs(runs))
            return
        cache_views, chunk_views = self._run_pairs(runs, dst)
        torch._foreach_copy_(chunk_views, cache_views)

    @override
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        """Copy the KV of tokens ``[start, end)`` from ``memory_obj`` (host or
        device) into the caches. Requires ``slot_mapping`` in ``kwargs``."""
        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            raise ValueError("slot_mapping is required for RBLN H2D")
        self._ensure_ready("H2D", kwargs.get("kvcaches"))
        src = self._chunk(memory_obj, end - start)
        # A host chunk is uploaded once and scattered on device; MLA rows are
        # contiguous, so they go straight from the host.
        if src.device != self._device and not (
            self.use_mla and src.device.type == "cpu"
        ):
            staged = self._staging_view(src.shape)
            staged.copy_(src)
            src = staged
        cache_views, chunk_views = self._run_pairs(
            self._runs(slot_mapping[start:end]), src
        )
        torch._foreach_copy_(cache_views, chunk_views)

    @override
    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs,
    ) -> None:
        for memory_obj, s, e in zip(memory_objs, starts, ends, strict=True):
            self.from_gpu(memory_obj, s, e, **kwargs)

    @override
    def batched_to_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj] | list[int] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs,
    ) -> None:
        if memory_objs is None or starts is None or ends is None:
            return
        for memory_obj, s, e in zip(memory_objs, starts, ends, strict=True):
            self.to_gpu(memory_obj, s, e, **kwargs)
