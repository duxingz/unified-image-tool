# -*- coding: utf-8 -*-
"""v1.22.2 — fix the first-launch brick: the single-instance lock used to
mistake "config dir does not exist yet" (QLockFile UnknownError) for "already
running", so on a fresh machine the FIRST double-click showed the「已经在运行」
box and the app could never open (save_config, the only mkdir, never ran)."""
from app.main import acquire_single_instance_lock


def test_first_launch_with_missing_config_dir(tmp_path):
    """THE reported bug: fresh machine, %APPDATA%\\AutomaticCoding2 absent.
    The lock must be acquired AND the dir created (config.json is written
    there later); it must NOT read as 'already running'."""
    fresh = tmp_path / "appdata" / "AutomaticCoding2"      # does NOT exist
    lock = acquire_single_instance_lock(fresh)
    assert lock is not None, "missing config dir must not read as running"
    assert fresh.is_dir() and (fresh / "app.lock").exists()


def test_second_instance_still_rejected(tmp_path):
    """A live lock is a LockFailedError — the only signal that means 'the
    app is already running'."""
    first = acquire_single_instance_lock(tmp_path)
    assert first is not None
    assert acquire_single_instance_lock(tmp_path) is None


def test_release_allows_fresh_acquire(tmp_path):
    lock1 = acquire_single_instance_lock(tmp_path)
    assert lock1 is not None
    del lock1                                   # released on GC
    assert acquire_single_instance_lock(tmp_path) is not None


def test_broken_dir_does_not_fake_running(tmp_path):
    """A config dir that cannot even be created (a FILE sits in the path)
    must start the app unenforced — never show the false dialog."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    lock = acquire_single_instance_lock(blocker / "nested" / "AutomaticCoding2")
    assert lock is not None, "I/O failure must not read as 'already running'"
