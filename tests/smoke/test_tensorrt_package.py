# SPDX-License-Identifier: Apache-2.0
"""The InProcess distribution owns its TensorRT package exports."""

# Standard
from types import ModuleType
import importlib
import sys

# Third Party
import pytest


def test_tensorrt_package_exports_inprocess_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package exports preserve the canonical adapter's object identities."""
    name = "lmcache.inprocess.integration.tensorrt_llm"
    adapter = ModuleType(f"{name}.tensorrt_adapter")
    exports = {
        "LMCacheKvConnectorScheduler": type("Scheduler", (), {}),
        "LMCacheKvConnectorWorker": type("Worker", (), {}),
        "destroy_engine": lambda: None,
    }
    for symbol, value in exports.items():
        setattr(adapter, symbol, value)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        package = importlib.import_module(name)
        assert set(package.__all__) == set(exports)
        for symbol, value in exports.items():
            assert getattr(package, symbol) is value
    finally:
        sys.modules.pop(name, None)
