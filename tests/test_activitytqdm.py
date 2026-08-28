# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``_ActivityTqdm`` must survive the exact instance calls huggingface_hub's xet path makes.

``_ActivityTqdm`` is a hand-written tqdm stand-in we hand to ``snapshot_download`` as its
``tqdm_class`` so a whole bundle's byte progress lands on one activity line instead of
tqdm's per-file bars. huggingface_hub then treats every bar we produce as a real tqdm and
calls tqdm *instance* members on it. #36 covered the class-level lock hooks
(``get_lock``/``set_lock``); this file pins the instance surface the xet reconstruction
reporter needs, which is where issue #41 crashed:

    ``_set_aggregate_rate_postfix(bar)``  ->  ``bar.format_dict.get("rate")``

evaluated on our bar, raising ``AttributeError: '_ActivityTqdm' object has no attribute
'format_dict'``. The tests below reproduce that call (and its siblings) directly, so they
fail on the pre-fix class and pass once the surface exists — no network, no real download.
"""

from tt_kernel.hub import _ActivityTqdm


def _byte_bar(total=1000, initial=0):
    """A bar built the way ``snapshot_download``'s ``reconstruct_progress`` is: byte unit,
    which is the only case ``_ActivityTqdm`` aggregates and the case xet reports rate on."""
    return _ActivityTqdm(total=total, initial=initial, unit="B", desc="Reconstructing")


def test_format_dict_get_rate_is_the_41_repro():
    """Verbatim shape of ``_set_aggregate_rate_postfix`` in
    huggingface_hub/utils/_xet_progress_reporting.py — the line that raised in #41."""
    bar = _byte_bar()
    # This is the whole crash: attribute access, then .get on the result.
    rate = bar.format_dict.get("rate")
    assert rate is None  # the reporter renders "???B/s" for a missing rate — that's fine


def test_format_dict_is_a_plain_dict_with_the_keys_tqdm_exposes():
    bar = _byte_bar(total=2048, initial=512)
    fd = bar.format_dict
    assert isinstance(fd, dict)
    # Keys real tqdm.format_dict carries that any caller might read back.
    for key in ("rate", "n", "total", "elapsed"):
        assert key in fd, key
    assert fd["n"] == 512
    assert fd["total"] == 2048


def test_set_postfix_str_accepts_the_reporters_refresh_kwarg():
    """``_set_aggregate_rate_postfix`` finishes with
    ``bar.set_postfix_str(<str>, refresh=False)`` — must be callable, must be a no-op."""
    bar = _byte_bar()
    assert bar.set_postfix_str("1.2MB/s", refresh=False) is None


def test_transfer_hooks_are_no_op_callables():
    """When this class advertises ``update_transfer`` the xet reporter collapses the
    transfer bar into the reconstruction bar and then calls ``update_transfer`` /
    ``set_transfer_postfix_str`` on this instance (utils/_xet_progress_reporting.py;
    file_download.py's http_get loop). Both must exist and no-op."""
    bar = _byte_bar()
    assert callable(getattr(bar, "update_transfer", None))
    assert bar.update_transfer(128) is None
    assert bar.set_transfer_postfix_str("3.4MB/s", refresh=False) is None


def test_aggregate_rate_postfix_end_to_end():
    """Drive the reporter's helper against our bar exactly as snapshot_download's
    ``_AggregatedTqdm.set_postfix_str`` does, so the two lines are exercised together."""
    bar = _byte_bar(total=4096)
    bar.update(1024)

    def _set_aggregate_rate_postfix(b):  # shape lifted from _xet_progress_reporting.py
        b.set_postfix_str(str(b.format_dict.get("rate")), refresh=False)

    _set_aggregate_rate_postfix(bar)  # raised AttributeError before the fix
    assert bar.n == 1024


def test_get_lock_still_works_regression():
    """#36 regression guard: the class-level lock hooks the thread_map path needs must
    keep working alongside the new instance surface."""
    lock = _ActivityTqdm.get_lock()
    assert lock is not None
    assert _ActivityTqdm.get_lock() is lock  # cached on the class, not re-created
    _ActivityTqdm.set_lock(lock)
    assert _ActivityTqdm.get_lock() is lock
