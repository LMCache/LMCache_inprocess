# SPDX-License-Identifier: Apache-2.0
"""Backoff of LMCacheAsyncLookupClient while lookups are in progress."""

# Standard
from unittest.mock import patch
import uuid

# Third Party
import pytest

# First Party
from lmcache.inprocess.v1.lookup_client import lmcache_async_lookup_client
from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)
from tests.inprocess.utils import create_test_config, create_test_metadata

BACKOFF = 0.01


class FakeClock:
    """Stands in for the time module: sleeping only moves the clock."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    fake = FakeClock()
    with patch.object(lmcache_async_lookup_client, "time", fake):
        yield fake


@pytest.fixture
def make_client():
    clients = []

    def _make(once_per_pass):
        extra_config = {"lookup_backoff_time": BACKOFF}
        if once_per_pass is not None:
            extra_config["lookup_backoff_once_per_pass"] = once_per_pass
        client = LMCacheAsyncLookupClient(
            create_test_config(extra_config=extra_config),
            create_test_metadata(engine_id=f"test_engine_{uuid.uuid4().hex[:8]}"),
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        # The response thread is blocked in recv() on the pull socket, and a
        # socket must not be closed under a thread that is using it. It is a
        # daemon thread: leave it and its socket to process exit.
        client.running = False
        for socket in client.push_sockets:
            socket.close(linger=0)


def _scheduling_pass(client, lookup_ids):
    """What the scheduler does in one pass: poll every waiting request."""
    return [client.lookup_cache(lookup_id) for lookup_id in lookup_ids]


def test_backoff_once_per_scheduling_pass(clock, make_client):
    client = make_client(once_per_pass=True)
    lookup_ids = [f"req-{i}" for i in range(5)]

    # The first poll of a request registers it and does not wait.
    assert _scheduling_pass(client, lookup_ids) == [-1] * 5
    assert clock.sleeps == []

    assert _scheduling_pass(client, lookup_ids) == [None] * 5
    assert clock.sleeps == [BACKOFF]

    # The engine did other work; the next pass waits again, once.
    clock.now += 0.005
    assert _scheduling_pass(client, lookup_ids) == [None] * 5
    assert clock.sleeps == [BACKOFF, BACKOFF]


def test_backoff_once_covers_new_lookups(clock, make_client):
    client = make_client(once_per_pass=True)

    for i in range(5):
        assert client.lookup([1, 2, 3], f"req-{i}") is None

    assert clock.sleeps == [BACKOFF]


@pytest.mark.parametrize("once_per_pass", [None, False])
def test_backoff_per_request_by_default(clock, make_client, once_per_pass):
    client = make_client(once_per_pass=once_per_pass)
    lookup_ids = [f"req-{i}" for i in range(5)]

    _scheduling_pass(client, lookup_ids)
    assert _scheduling_pass(client, lookup_ids) == [None] * 5

    assert clock.sleeps == [BACKOFF] * 5


def test_finished_lookup_is_returned_without_waiting(clock, make_client):
    client = make_client(once_per_pass=True)
    _scheduling_pass(client, ["req-0", "req-1"])
    with client.lock:
        client.reqs_status["req-1"] = 512

    assert _scheduling_pass(client, ["req-0", "req-1"]) == [None, 512]
    assert clock.sleeps == [BACKOFF]
