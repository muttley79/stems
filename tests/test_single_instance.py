"""Single-instance guard tests (loopback-socket based).

Each test pins ``STEMS_GUI_PORT`` to a free ephemeral port so runs don't collide
with a real GUI or with each other.
"""

import socket
import time

import pytest

from stems.gui import single_instance as si


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def port(monkeypatch):
    p = _free_port()
    monkeypatch.setenv("STEMS_GUI_PORT", str(p))
    return p


def test_first_instance_is_primary(port):
    server = si.acquire_or_signal()
    try:
        assert server is not None
        assert server.is_primary is True
    finally:
        server.close()


def test_second_launch_signals_first_and_exits(port):
    first = si.acquire_or_signal()
    assert first is not None and first.is_primary
    first.start()
    try:
        # A second launch finds the port taken, pings the first, and bows out.
        second = si.acquire_or_signal()
        assert second is None

        # The running instance receives exactly one show request.
        got = False
        for _ in range(100):
            if first.consume_show_request():
                got = True
                break
            time.sleep(0.01)
        assert got, "primary never received the show request"
        assert first.consume_show_request() is False  # consumed once
    finally:
        first.close()


def test_port_freed_after_close(port):
    first = si.acquire_or_signal()
    assert first.is_primary
    first.close()
    # Once closed, a fresh launch can bind the port again as primary.
    second = si.acquire_or_signal()
    try:
        assert second.is_primary is True
    finally:
        second.close()


def test_fail_open_when_port_held_by_foreigner(port):
    # A non-stems listener holds the port: connecting succeeds but the protocol
    # handshake gets no answer, so we run anyway (not primary, no relaying).
    foreign = socket.socket()
    foreign.bind(("127.0.0.1", port))
    foreign.listen(1)
    try:
        server = si.acquire_or_signal()
        assert server is not None
        assert server.is_primary is False
    finally:
        if server:
            server.close()
        foreign.close()
