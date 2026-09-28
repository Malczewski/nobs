from __future__ import annotations

from pathlib import Path

from app.storage import SeenStore


def _store(tmp_path: Path) -> SeenStore:
    return SeenStore(str(tmp_path / "nested" / "seen.db"))


def test_creates_parent_directory(tmp_path: Path):
    db_path = tmp_path / "nested" / "seen.db"
    assert not db_path.parent.exists()
    SeenStore(str(db_path))
    assert db_path.parent.exists()


def test_unseen_item_is_not_seen(tmp_path: Path):
    store = _store(tmp_path)
    assert store.is_seen("rss", "item-1") is False


def test_mark_seen_then_is_seen(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_seen("rss", "item-1")
    assert store.is_seen("rss", "item-1") is True


def test_namespaces_are_isolated(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_seen("rss", "same-id")
    assert store.is_seen("tg_monitor", "same-id") is False


def test_mark_seen_is_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_seen("rss", "item-1")
    store.mark_seen("rss", "item-1")  # must not raise (PRIMARY KEY conflict)
    assert store.is_seen("rss", "item-1") is True


def test_mark_seen_many(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_seen_many("rss", ["a", "b", "c"])
    assert store.is_seen("rss", "a")
    assert store.is_seen("rss", "b")
    assert store.is_seen("rss", "c")
    assert store.is_seen("rss", "d") is False


def test_mark_seen_many_empty_list_is_a_noop(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_seen_many("rss", [])  # must not raise on an empty executemany


def test_persists_across_reopen(tmp_path: Path):
    db_path = tmp_path / "seen.db"
    store = SeenStore(str(db_path))
    store.mark_seen("rss", "item-1")
    store.close()

    reopened = SeenStore(str(db_path))
    assert reopened.is_seen("rss", "item-1") is True
