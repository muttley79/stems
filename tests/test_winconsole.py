"""Tests for the child-console suppression patch.

The patch is process-global (it replaces ``subprocess.Popen.__init__``), so each
test saves and restores that, plus the module's ``_installed`` flag, in a
``finally`` to keep the rest of the suite unaffected.
"""

import subprocess
import sys

import pytest

from stems.gui import _winconsole


def test_suppress_is_noop_off_windows():
    if sys.platform.startswith("win"):
        pytest.skip("covered by the Windows-specific test")
    saved = subprocess.Popen.__init__
    _winconsole._installed = False
    try:
        _winconsole.suppress_child_consoles()
        assert subprocess.Popen.__init__ is saved  # nothing patched off Windows
    finally:
        subprocess.Popen.__init__ = saved
        _winconsole._installed = False


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="console suppression is Windows-only"
)
def test_suppress_patches_and_children_still_run():
    saved = subprocess.Popen.__init__
    _winconsole._installed = False
    try:
        _winconsole.suppress_child_consoles()
        # The class was actually patched...
        assert subprocess.Popen.__init__ is not saved
        # ...and children still launch and return output correctly under it.
        out = subprocess.run(
            [sys.executable, "-c", "print('ok')"], capture_output=True, text=True
        )
        assert out.stdout.strip() == "ok"
        # Idempotent: a second call doesn't re-wrap.
        twice = subprocess.Popen.__init__
        _winconsole.suppress_child_consoles()
        assert subprocess.Popen.__init__ is twice
    finally:
        subprocess.Popen.__init__ = saved
        _winconsole._installed = False
