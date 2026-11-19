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


class EICConnectorAdapter(ConnectorAdapter):
    """Adapter for EIC connectors."""

    def __init__(self) -> None:
        super().__init__("eic://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .eic_connector import EICConnector

        logger.info("Creating EIC connector for URL: %s", context.url)
        return EICConnector(
            endpoint=context.url,
            loop=context.loop,
            memory_allocator=context.local_cpu_backend,
        )
