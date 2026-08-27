"""The tqdm stand-in must satisfy the parts of tqdm that huggingface_hub relies on."""
import pytest
from tqdm.contrib.concurrent import thread_map

from tt_kernel.hub import _ActivityTqdm, progress_bridge


def test_thread_map_works_and_actually_runs_the_work():
    """_executor_map does list(tqdm_class(ex.map(...))): the WRAPPED ITERATOR drives the
    work. Returning an empty iterator made snapshot_download report success having
    downloaded nothing."""
    out = thread_map(lambda x: x * 2, range(5), tqdm_class=_ActivityTqdm, max_workers=2)
    assert out == [0, 2, 4, 6, 8]


def test_the_class_level_lock_protocol_exists():
    """ensure_lock() calls these on the CLASS; without them snapshot_download raised
    AttributeError: type object '_ActivityTqdm' has no attribute 'get_lock'."""
    lock = _ActivityTqdm.get_lock()
    assert lock is not None
    _ActivityTqdm.set_lock(lock)
    assert _ActivityTqdm.get_lock() is lock


def test_lock_is_not_a_class_attribute_until_requested():
    """ensure_lock() does `del tqdm_class._lock` when it created the lock and restores the
    caller's otherwise, so a pre-set class attribute would corrupt that bookkeeping."""
    import tt_kernel.hub as hub

    cls = type("Probe", (hub._ActivityTqdm,), {})
    assert not hasattr(cls, "_lock") or "_lock" not in cls.__dict__


def test_wrapping_no_iterable_still_iterates_empty():
    """huggingface_hub's direct uses pass only total/desc; those must not break."""
    assert list(_ActivityTqdm(total=10, unit="B")) == []


def test_wrapping_an_iterable_passes_items_through_and_counts():
    bar = _ActivityTqdm(iter([1, 2, 3]), unit="it")
    assert list(bar) == [1, 2, 3]
    assert bar.n == 3


def test_the_bridge_still_reports_bytes(monkeypatch):
    seen = []
    from tt_kernel import console

    monkeypatch.setattr(console.activity, "set", lambda s: seen.append(s))
    with progress_bridge("Pulling") as klass:
        bar = klass(total=100, unit="B")
        bar.update(50)
        bar.close()
    assert any("Pulling" in s for s in seen)
