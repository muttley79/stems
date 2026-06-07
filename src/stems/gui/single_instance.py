"""Single-instance guard + activation for the desktop GUI.

The first GUI process binds a loopback TCP socket on a fixed port. That bind
doubles as the lock (a second process cannot bind the same port) *and* as a tiny
IPC channel: when a later launch finds the port taken, it connects and sends a
one-line "show" request; the running instance acknowledges and raises its window
to the front, then the second process exits.

Loopback-only (``127.0.0.1``) so it never prompts the firewall and is not
reachable off the machine. The OS frees the port when the process ends (even on
a crash), so there is nothing stale to clean up, and no extra dependency.

Override the port with the ``STEMS_GUI_PORT`` environment variable.
"""

from __future__ import annotations

import os
import socket
import threading

_HOST = "127.0.0.1"
_DEFAULT_PORT = 49327
_MAGIC = b"STEMS-GUI-SHOW"
_OK = b"OK"


def _port() -> int:
    raw = os.environ.get("STEMS_GUI_PORT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return _DEFAULT_PORT


class SingleInstanceServer:
    """Owns the loopback lock socket and relays activation requests.

    A ``None`` socket means we are *not* the lock owner (a foreign service held
    the port); the app still runs, just without activation relaying. Use
    :meth:`consume_show_request` from the UI thread to learn when another launch
    asked this window to come forward.
    """

    def __init__(self, sock: socket.socket | None) -> None:
        self._sock = sock
        self._show_requested = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_primary(self) -> bool:
        """True if this process owns the single-instance lock socket."""
        return self._sock is not None

    def start(self) -> None:
        """Begin accepting activation requests on a daemon thread."""
        if self._sock is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def consume_show_request(self) -> bool:
        """Return True at most once per request that another instance pinged us.

        Thread-safe (backed by :class:`threading.Event`); poll it from the UI
        thread so the actual window raise happens on the main thread.
        """
        if self._show_requested.is_set():
            self._show_requested.clear()
            return True
        return False

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _serve(self) -> None:
        sock = self._sock
        if sock is None:
            return
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return  # socket closed → shutting down
            with conn:
                try:
                    data = conn.recv(64)
                except OSError:
                    continue
                if data.strip() == _MAGIC:
                    try:
                        conn.sendall(_OK + b"\n")
                    except OSError:
                        pass
                    self._show_requested.set()


def acquire_or_signal() -> SingleInstanceServer | None:
    """Become the primary instance, or signal the existing one and bow out.

    Returns a :class:`SingleInstanceServer` to keep for the app's lifetime when
    this process should run (it is primary, or a foreign service holds the port).
    Returns ``None`` when an already-running stems GUI was asked to come to the
    front - the caller should simply exit.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((_HOST, _port()))
        sock.listen(1)
        return SingleInstanceServer(sock)
    except OSError:
        sock.close()

    # Port busy: try to wake the running stems instance.
    if _signal_existing():
        return None
    # Held by something that isn't us - run anyway (fail open), no relaying.
    return SingleInstanceServer(None)


def _signal_existing() -> bool:
    """Ping the running instance to show itself. True if it acknowledged."""
    try:
        with socket.create_connection((_HOST, _port()), timeout=1.0) as c:
            c.sendall(_MAGIC + b"\n")
            c.settimeout(1.0)
            reply = c.recv(16)
        return reply.strip() == _OK
    except OSError:
        return False
