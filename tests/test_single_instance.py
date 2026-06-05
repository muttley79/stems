"""Single-instance lock tests.

OS advisory locks taken on two *separate* opens of the same file conflict even
within one process (Windows byte-range locks are per-handle; POSIX ``flock`` is
per open file description), so a second ``acquire`` in the same test stands in
for a second process and must be refused.
"""

import pytest

from stems.gui import single_instance as si


@pytest.fixture(autouse=True)
def _reset_lock():
    si._lock_handle = None
    yield
    si.release_single_instance_lock()


def test_first_instance_acquires(tmp_path):
    assert si.acquire_single_instance_lock("test-app", tmp_path) is True
    assert si._lock_handle is not None


def test_second_instance_is_blocked(tmp_path):
    assert si.acquire_single_instance_lock("test-app", tmp_path) is True
    # A second acquisition (a stand-in for another process) is refused.
    assert si.acquire_single_instance_lock("test-app", tmp_path) is False


def test_lock_reacquirable_after_release(tmp_path):
    assert si.acquire_single_instance_lock("test-app", tmp_path) is True
    si.release_single_instance_lock()
    assert si._lock_handle is None
    # Once released, a fresh launch can take it again.
    assert si.acquire_single_instance_lock("test-app", tmp_path) is True
