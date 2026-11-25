# SPDX-License-Identifier: Apache-2.0
"""Tests for the controller manager's ``*_socket_has_pending`` gauges.

ZMQ sockets are not thread-safe. The gauges are read from whichever thread
serves the metrics scrape, so they must report a value recorded by the receive
loops instead of querying the sockets themselves.
"""

# Standard
from collections.abc import Iterator
import asyncio
import threading

# Third Party
import pytest
import zmq

# First Party
from lmcache.inprocess.v1.cache_controller.controller_manager import (
    LMCacheControllerManager,
)
from lmcache.inprocess.v1.cache_controller.observability import PrometheusLogger


class _RecordingSocket:
    """Stand-in for a ZMQ socket that records which threads query it.

    ``recv_multipart`` returns the queued batches in order, then raises
    ``asyncio.CancelledError`` to end the receive loop. ``get`` returns the
    queued event masks in order (0 once exhausted) and records the calling
    thread.
    """

    def __init__(
        self, batches: list[list[bytes]] | None = None, events: list[int] | None = None
    ) -> None:
        self._batches = list(batches or [])
        self._events = list(events or [])
        self.get_thread_ids: list[int] = []

    async def recv_multipart(self) -> list[bytes]:
        """Return the next queued batch, or cancel once all are consumed."""
        if not self._batches:
            raise asyncio.CancelledError
        return self._batches.pop(0)

    def get(self, option: int) -> int:
        """Return the next queued event mask and record the calling thread."""
        self.get_thread_ids.append(threading.get_ident())
        return self._events.pop(0) if self._events else 0


def _gauge_value(gauge) -> float:
    """Read a gauge the way a metrics scrape does."""
    return gauge.collect()[0].samples[0].value


def _read_pending_gauges_in_thread() -> tuple[float, float]:
    """Read both has_pending gauges from a separate thread, like a scrape."""
    prometheus_logger = PrometheusLogger.GetInstance()
    values: list[float] = []

    def scrape() -> None:
        values.append(_gauge_value(prometheus_logger.pull_socket_has_pending))
        values.append(_gauge_value(prometheus_logger.reply_socket_has_pending))

    thread = threading.Thread(target=scrape)
    thread.start()
    thread.join()
    return values[0], values[1]


@pytest.fixture
def manager() -> Iterator[LMCacheControllerManager]:
    """Create a controller manager bound to OS-assigned ports."""
    controller_manager = LMCacheControllerManager(
        controller_urls={"pull": "127.0.0.1:0", "reply": "127.0.0.1:0"},
        health_check_interval=0,
        lmcache_worker_timeout=30,
    )
    pull_socket = controller_manager.controller_pull_socket
    reply_socket = controller_manager.controller_reply_socket

    yield controller_manager

    # Tests swap in recording sockets; close the real ones.
    controller_manager.controller_pull_socket = pull_socket
    controller_manager.controller_reply_socket = reply_socket
    controller_manager.close()


def test_scrape_does_not_touch_sockets(manager: LMCacheControllerManager) -> None:
    """Reading the gauges from another thread never queries a socket."""
    pull_socket = _RecordingSocket()
    reply_socket = _RecordingSocket()
    manager.controller_pull_socket = pull_socket
    manager.controller_reply_socket = reply_socket

    assert _read_pending_gauges_in_thread() == (0, 0)

    assert pull_socket.get_thread_ids == []
    assert reply_socket.get_thread_ids == []


def test_pull_loop_refreshes_pending_flag(manager: LMCacheControllerManager) -> None:
    """The PULL loop records the backlog on its own thread after each receive."""
    # One message still queued after the first receive.
    pull_socket = _RecordingSocket(
        batches=[[]],
        events=[zmq.POLLIN],  # type: ignore[attr-defined]
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.handle_batched_push_request(pull_socket))

    assert manager.pull_socket_has_pending == 1
    assert _read_pending_gauges_in_thread() == (1, 0)
    assert pull_socket.get_thread_ids == [threading.get_ident()]


def test_reply_loop_refreshes_pending_flag(manager: LMCacheControllerManager) -> None:
    """The ROUTER loop records the backlog on its own thread after each receive."""
    # Frames shorter than [identity, empty, payload] are logged and skipped.
    reply_socket = _RecordingSocket(
        batches=[[b"id"]],
        events=[zmq.POLLIN],  # type: ignore[attr-defined]
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.handle_batched_req_request(reply_socket))

    assert manager.reply_socket_has_pending == 1
    assert _read_pending_gauges_in_thread() == (0, 1)
    assert reply_socket.get_thread_ids == [threading.get_ident()]
