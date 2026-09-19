from opensysone.cache import DecisionCache, cache_key


def test_key_is_stable_and_order_independent_for_dicts():
    assert cache_key("s", {"a": 1, "b": 2}) == cache_key("s", {"b": 2, "a": 1})
    assert cache_key("s", {"a": 1}) != cache_key("t", {"a": 1})
    assert len(cache_key("x")) == 64


def test_lru_evicts_oldest():
    c = DecisionCache(max_entries=2)
    c.put("a", {"v": 1})
    c.put("b", {"v": 2})
    assert c.get("a") == {"v": 1}  # touch a, b becomes oldest
    c.put("c", {"v": 3})
    assert c.get("b") is None
    assert c.get("a") == {"v": 1}
    assert len(c) == 2


def test_sqlite_persists_across_instances(tmp_path):
    p = str(tmp_path / "cache.sqlite")
    DecisionCache(max_entries=10, path=p).put("k", {"answer": 42})
    assert DecisionCache(max_entries=10, path=p).get("k") == {"answer": 42}


def test_clear_empties_memory_and_disk(tmp_path):
    p = str(tmp_path / "cache.sqlite")
    c = DecisionCache(path=p)
    c.put("k", {"v": 1})
    c.clear()
    assert c.get("k") is None
    assert DecisionCache(path=p).get("k") is None
