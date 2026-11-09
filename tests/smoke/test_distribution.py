# SPDX-License-Identifier: Apache-2.0
# Standard
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import files

# Third Party
import torch

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.gpu_connector import CreateGPUConnector
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.inprocess.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.utils import EngineType


def test_distribution_metadata() -> None:
    """The installed wheel exposes its own distribution metadata."""
    try:
        installed = distribution("lmcache_inprocess")
    except PackageNotFoundError:
        # Source-tree tests may run before installation. Wheel CI always takes
        # the installed branch above.
        return
    assert installed.metadata["Name"] == "lmcache_inprocess"
    assert any(
        requirement.startswith("lmcache>=") for requirement in installed.requires
    )


def test_frontend_assets_are_packaged() -> None:
    """The controller frontend remains usable from an installed wheel."""
    static_root = files("lmcache.inprocess.v1.cache_controller.frontend") / "static"
    assert (static_root / "index.html").is_file()
    assert (static_root / "js" / "controller_app.js").is_file()


def test_console_script_entrypoints_import() -> None:
    """Both console scripts import against the installed LMCache release.

    The entry modules pull upstream helpers, so a symbol that LMCache renamed
    or removed breaks ``lmcache-inprocess-standalone --help`` while the rest of
    the wheel stays importable. Importing them here keeps that visible in the
    smoke suite and in the nightly LMCache ``dev`` compatibility job.
    """
    for module in (
        "lmcache.inprocess.standalone.__main__",
        "lmcache.inprocess.v1.api_server.__main__",
    ):
        assert import_module(module) is not None


def test_mock_connector_end_to_end_factory() -> None:
    """Construct the in-process connector through its public factory."""
    metadata = LMCacheMetadata(
        model_name="ci-smoke",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(2, 2, 32, 2, 8),
    )
    connector = CreateGPUConnector(
        LMCacheEngineConfig.from_defaults(), metadata, EngineType.MOCK
    )
    assert isinstance(connector, GPUConnectorInterface)
    assert isinstance(connector, MockGPUConnector)
    assert connector.get_shape(16) == torch.Size([2, 2, 16, 2, 8])
