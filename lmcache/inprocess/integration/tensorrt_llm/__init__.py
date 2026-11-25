# SPDX-License-Identifier: Apache-2.0
"""LMCache in-process integration for NVIDIA TensorRT-LLM.

The LMCache engine runs inside the TensorRT-LLM process. Select the ``lmcache``
preset, or configure the adapter explicitly::

    kv_connector_config:
      connector_module: lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter
      connector_scheduler_class: LMCacheKvConnectorScheduler
      connector_worker_class: LMCacheKvConnectorWorker

TensorRT-LLM is optional; package exports are empty when its adapter cannot be
imported.
"""

try:
    # First Party
    from lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter import (
        LMCacheKvConnectorScheduler,
        LMCacheKvConnectorWorker,
        destroy_engine,
    )

    __all__ = [
        "LMCacheKvConnectorScheduler",
        "LMCacheKvConnectorWorker",
        "destroy_engine",
    ]
except ImportError:
    __all__ = []
