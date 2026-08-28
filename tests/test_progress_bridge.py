# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The activity bridge is a REAL tqdm with its output redirected.

It used to imitate tqdm, and huggingface_hub kept reaching for parts that were missing —
get_lock/set_lock, the wrapped-iterable protocol, format_dict — each surfacing as an
AttributeError in someone's download. These tests pin the properties that made that class
of bug possible, so an imitation cannot be reintroduced quietly.
"""

import io
import sys

import pytest
from tqdm.contrib.concurrent import thread_map

from tt_kernel.hub import _ActivityTqdm, progress_bridge


def test_it_is_a_real_tqdm():
    """The structural fix: every attribute exists because it IS tqdm."""
    from tqdm import tqdm

    assert issubclass(_ActivityTqdm, tqdm)


@pytest.mark.parametrize("attr", [
    # touched by huggingface_hub's xet progress reporter and _snapshot_download
    # (format_desc is a method on hub's own reporter, not on the bar)
    "format_dict", "set_postfix_str", "set_description", "refresh",
    "update", "close", "n", "total", "get_lock", "set_lock",
])
def test_the_surface_huggingface_hub_touches_exists(attr):
    bar = _ActivityTqdm(total=10, unit="B")
    try:
        assert hasattr(bar, attr), attr
    finally:
        bar.close()


def test_format_dict_carries_a_rate():
    """_set_aggregate_rate_postfix does bar.format_dict.get("rate") — the AttributeError
    a consumer hit mid-pull."""
    bar = _ActivityTqdm(total=100, unit="B")
    try:
        bar.update(10)
        d = bar.format_dict
        assert "rate" in d and "n" in d and "total" in d
    finally:
        bar.close()


def test_set_postfix_str_with_refresh_false_is_accepted():
    """The exact call from _xet_progress_reporting."""
    bar = _ActivityTqdm(total=100, unit="B")
    try:
        bar.set_postfix_str("12.3 MB/s", refresh=False)
    finally:
        bar.close()


def test_thread_map_runs_the_work():
    """_executor_map does list(tqdm_class(ex.map(...))): the wrapped iterator IS the work,
    and it needs the class-level lock protocol to get that far."""
    assert thread_map(lambda x: x * 2, range(5), tqdm_class=_ActivityTqdm,
                      max_workers=2) == [0, 2, 4, 6, 8]


def test_wrapping_an_iterable_passes_items_through():
    bar = _ActivityTqdm(iter([1, 2, 3]), unit="it")
    assert list(bar) == [1, 2, 3]
    assert bar.n == 3


def test_nothing_reaches_the_terminal(capsys, monkeypatch):
    """tqdm renders into a sink; our one activity line is the only output."""
    from tt_kernel import console

    monkeypatch.setattr(console.activity, "set", lambda s: None)
    bar = _ActivityTqdm(total=100, unit="B")
    bar.update(50)
    bar.refresh()
    bar.close()
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_byte_totals_are_aggregated_across_concurrent_bars(monkeypatch):
    seen = []
    from tt_kernel import console

    monkeypatch.setattr(console.activity, "set", lambda s: seen.append(s))
    _ActivityTqdm._live.clear()
    a = _ActivityTqdm(total=100, unit="B")
    b = _ActivityTqdm(total=100, unit="B")
    try:
        a.update(30)
        b.update(70)
        assert seen and "100" in seen[-1].replace(",", "")
    finally:
        a.close()
        b.close()


def test_a_non_byte_bar_does_not_corrupt_the_total(monkeypatch):
    """A "Fetching 5 files" bar counts files, not bytes."""
    from tt_kernel import console

    monkeypatch.setattr(console.activity, "set", lambda s: None)
    files = _ActivityTqdm(total=5, unit="it")
    try:
        files.update(3)
        assert id(files) not in _ActivityTqdm._live
    finally:
        files.close()


def test_closing_removes_the_bar_from_the_aggregate(monkeypatch):
    from tt_kernel import console

    monkeypatch.setattr(console.activity, "set", lambda s: None)
    bar = _ActivityTqdm(total=100, unit="B")
    bar.update(10)
    key = id(bar)
    bar.close()
    assert key not in _ActivityTqdm._live


def test_the_bridge_labels_the_activity(monkeypatch):
    seen = []
    from tt_kernel import console

    monkeypatch.setattr(console.activity, "set", lambda s: seen.append(s))
    with progress_bridge("Pulling") as klass:
        bar = klass(total=100, unit="B")
        bar.update(50)
        bar.close()
    assert any("Pulling" in s for s in seen)


def test_display_returns_true_like_tqdms_own():
    """tqdm's display() returns True and close() DEPENDS on it:

        if self.display(msg='', pos=pos) and not pos:

    so an override returning None would change close()'s behaviour. Delegating to
    super().display() would also format a bar and write it to the sink — work whose only
    purpose is to be discarded.
    """
    import inspect

    from tqdm.std import tqdm

    assert inspect.getsource(tqdm.display).rstrip().endswith("return True")

    bar = _ActivityTqdm(total=10, unit="B")
    try:
        assert bar.display() is True
    finally:
        bar.close()
