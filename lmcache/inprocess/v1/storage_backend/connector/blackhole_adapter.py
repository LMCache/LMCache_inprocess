# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.inprocess.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.inprocess.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

try:
    # First Party
    from lmcache.logging import init_logger
except ModuleNotFoundError as exc:
    if exc.name not in {"lmcache.logging"}:
        raise
    # First Party
    from lmcache.v1.common.logging import init_logger

logger = init_logger(__name__)


class BlackholeConnectorAdapter(ConnectorAdapter):
    """Adapter for Blackhole connectors (for testing)."""

    def __init__(self) -> None:
        super().__init__("blackhole://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .blackhole_connector import BlackholeConnector

        logger.info("Creating Blackhole connector for URL: %s", context.url)
        return BlackholeConnector()
