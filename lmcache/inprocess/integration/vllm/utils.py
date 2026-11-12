# SPDX-License-Identifier: Apache-2.0
"""vLLM helpers that construct the legacy in-process engine."""

# Standard
from typing import TYPE_CHECKING, Any, Optional, Tuple
import os
import re
import threading

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig, load_ec_engine_config
from lmcache.inprocess.v1.config_base import apply_remote_configs, fetch_remote_config
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    calculate_draft_layers,
    extract_mm_features,
    extract_request_configs_from_sampling_params,
    get_size_bytes,
    is_dp_rank0,
    mla_enabled,
    vllm_layout_hints,
)
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

__all__ = [
    "ENGINE_NAME",
    "apply_mm_hashes_to_token_ids",
    "calculate_draft_layers",
    "calculate_local_rank_and_world_size",
    "extract_mm_features",
    "extract_request_configs_from_sampling_params",
    "get_size_bytes",
    "is_dp_rank0",
    "mla_enabled",
    "vllm_layout_hints",
]

logger = init_logger(__name__)

_config_instance: Optional[LMCacheEngineConfig] = None
_config_lock = threading.Lock()


def _vllm_local_device_index(parallel_config: Any) -> int:
    """Return the device index vLLM binds this worker to on a single node.

    Each vLLM data-parallel rank is its own engine whose TP/PCP/PP world
    starts at rank 0, so ``parallel_config.rank`` alone repeats on every
    data-parallel rank. This mirrors ``Worker.init_device`` in vLLM: offset
    by the local data-parallel rank, then map through the platform's
    logical-to-visible device mapping.
    """
    local_rank = parallel_config.rank
    if (
        parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
        and getattr(parallel_config, "data_parallel_backend", None) != "ray"
        and getattr(parallel_config, "nnodes_within_dp", 1) == 1
    ):
        dp_local_rank = parallel_config.data_parallel_rank_local
        if dp_local_rank is None:
            dp_local_rank = getattr(
                parallel_config,
                "data_parallel_index",
                parallel_config.data_parallel_rank,
            )
        local_rank += dp_local_rank * parallel_config.world_size

    try:
        # Third Party
        from vllm.platforms import current_platform
    except ImportError:
        return local_rank
    to_visible = getattr(
        current_platform, "logical_device_id_to_visible_device_id", None
    )
    return to_visible(local_rank) if to_visible is not None else local_rank


def calculate_local_rank_and_world_size(vllm_config: "VllmConfig") -> Tuple[int, int]:
    """Calculate the local worker id and local world size.

    Owned by this repository rather than imported from
    ``lmcache.integration.vllm.utils``: that helper only ever served the
    in-process engine, and upstream dropped it in LMCache#6412 ("Remove stale
    InProcess tests"). Importing the name made every entrypoint fail as soon as
    a newer LMCache was installed, because the symbol no longer exists there.

    Current assumption (TODO: add custom logic in the future):
    - Tensor Parallel is intra-node
    - Pipeline Parallel is inter-node

    Returns:
        Tuple[int, int]: (local_worker_id, local_world_size)
    """
    # First Party
    from lmcache import torch_dev

    parallel_config = vllm_config.parallel_config
    global_rank = parallel_config.rank
    global_world_size = parallel_config.world_size
    num_gpus = torch_dev.device_count()
    if global_world_size <= num_gpus:
        # single node case
        return _vllm_local_device_index(parallel_config), global_world_size
    else:
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        local_world_size = global_world_size // pp_size
        assert local_world_size == tp_size, (
            "LMCache is operating under the assumption that the "
            "local world size is equal to the tensor parallel size "
            "in multi-node deployment."
        )
        local_worker_id = global_rank % local_world_size
        return local_worker_id, local_world_size


def lmcache_get_or_create_config() -> LMCacheEngineConfig:
    """Load and cache the configuration for the in-process engine."""
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                if "LMCACHE_CONFIG_FILE" not in os.environ:
                    logger.warning(
                        "No LMCache configuration file is set. Trying to read "
                        "configurations from environment variables."
                    )
                    _config_instance = LMCacheEngineConfig.from_env()
                    _config_instance.validate()
                else:
                    config_file = os.environ["LMCACHE_CONFIG_FILE"]
                    logger.info("Loading LMCache config file %s", config_file)
                    _config_instance = LMCacheEngineConfig.from_file(config_file)
                    _config_instance.update_config_from_env()

                remote_config_url = _config_instance.remote_config_url
                if remote_config_url:
                    remote_response = fetch_remote_config(
                        remote_config_url,
                        _config_instance.app_id,
                        _config_instance,
                    )
                    if remote_response:
                        _config_instance = apply_remote_configs(
                            _config_instance, remote_response
                        )
                    else:
                        logger.warning(
                            "Failed to fetch remote configuration from %s. "
                            "Using local configuration only.",
                            remote_config_url,
                        )
    return _config_instance


def create_lmcache_ec_config() -> LMCacheEngineConfig:
    """Create EC config from LMCache config plus EC-specific overrides."""
    return load_ec_engine_config(base_config=lmcache_get_or_create_config())


def create_lmcache_metadata(
    vllm_config=None,
    model_config=None,
    parallel_config=None,
    cache_config=None,
    role=None,
):
    """Create in-process engine metadata from vLLM configuration."""
    try:
        # Third Party
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    except ImportError:
        # Third Party
        from vllm.utils import get_kv_cache_torch_dtype

    config = lmcache_get_or_create_config()
    if vllm_config is not None:
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        cache_cfg = vllm_config.cache_config
    else:
        model_cfg = model_config
        parallel_cfg = parallel_config
        cache_cfg = cache_config

    kv_dtype = get_kv_cache_torch_dtype(cache_cfg.cache_dtype, model_cfg.dtype)
    use_mla = mla_enabled(model_cfg)
    kv_shape = (
        model_cfg.get_num_layers(parallel_cfg),
        1 if use_mla else 2,
        config.chunk_size,
        model_cfg.get_num_kv_heads(parallel_cfg),
        model_cfg.get_head_size(),
    )

    engine_id = None
    kv_connector_extra_config = None
    if vllm_config is not None and hasattr(vllm_config, "kv_transfer_config"):
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            engine_id = getattr(kv_transfer_config, "engine_id", None)
            kv_connector_extra_config = getattr(
                kv_transfer_config, "kv_connector_extra_config", None
            )

    metadata = LMCacheMetadata(
        model_name=model_cfg.model,
        world_size=parallel_cfg.world_size,
        local_world_size=parallel_cfg.world_size,
        worker_id=parallel_cfg.rank,
        local_worker_id=parallel_cfg.rank,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role=role,
        served_model_name=model_cfg.served_model_name,
        engine_id=engine_id,
        kv_connector_extra_config=kv_connector_extra_config,
    )
    return metadata, config


def validate_mla_config(config: LMCacheEngineConfig, use_mla: bool) -> None:
    """Validate MLA-related in-process configuration."""
    if use_mla and config.remote_serde not in ("naive", None):
        raise ValueError("MLA only works with naive serde mode..")
    if use_mla and config.use_layerwise and config.enable_blending:
        raise ValueError(
            "We haven't supported MLA with Cacheblend yet. Please disable blending."
        )


def set_dp_rank_controller_identity(config, vllm_config) -> None:
    """Give each vLLM data-parallel engine its own controller identity.

    Every DP engine's TP world starts at rank 0, so without this all DP
    engines on a host bind ``lmcache_worker_ports[worker_id]`` (the same
    port) and, with a configured ``lmcache_instance_id``, register the same
    ``(instance_id, worker_id)`` with the controller. Each DP engine now uses
    ``<lmcache_instance_id>-dp<rank>`` and the ports starting at index
    ``local_dp_rank * world_size`` (the list is rotated, so the per-worker
    port selection in the cache controller worker is unchanged). Nothing
    changes without data parallelism.
    """
    parallel_config = vllm_config.parallel_config
    dp_size = getattr(parallel_config, "data_parallel_size", 1) or 1
    if dp_size <= 1 or not config.lmcache_instance_id:
        return
    ktc = getattr(vllm_config, "kv_transfer_config", None)
    # vLLM appends _dp<rank> to engine_id for DP engines (MultiConnector
    # passes it on to its children).
    match = re.search(r"_dp(\d+)$", str(getattr(ktc, "engine_id", "") or ""))
    dp_rank = int(match.group(1)) if match else parallel_config.data_parallel_rank
    suffix = f"-dp{dp_rank}"
    if config.lmcache_instance_id.endswith(suffix):
        return  # already applied in this process (the config is a singleton)
    config.lmcache_instance_id = f"{config.lmcache_instance_id}{suffix}"
    ports = list(config.lmcache_worker_ports or [])
    if not ports:
        return
    local_dp_rank = getattr(parallel_config, "data_parallel_rank_local", None)
    if local_dp_rank is None:
        local_dp_rank = dp_rank
    world_size = parallel_config.world_size
    local_dp_size = (
        getattr(parallel_config, "data_parallel_size_local", None) or dp_size
    )
    if len(ports) < local_dp_size * world_size:
        logger.warning(
            "lmcache_worker_ports has %d port(s) for %d data-parallel engine(s) "
            "x %d worker(s) on this host; ports will be shared",
            len(ports),
            local_dp_size,
            world_size,
        )
    start = (local_dp_rank * world_size) % len(ports)
    config.lmcache_worker_ports = ports[start:] + ports[:start]
