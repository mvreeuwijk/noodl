from noodl._unionfind import UnionFind


def test_find_adds_unknown_items_as_singletons():
    uf = UnionFind()
    assert "a" not in uf
    assert uf.find("a") == "a" and "a" in uf


def test_union_merges_and_reports_whether_it_did():
    uf = UnionFind("abcd")
    assert uf.union("a", "b") and uf.union("c", "d") and uf.union("b", "d")
    assert not uf.union("a", "c")
    assert len({uf.find(x) for x in "abcd"}) == 1


def test_the_first_argument_root_is_kept():
    uf = UnionFind()
    uf.union("x", "y")
    assert uf.find("y") == "x"


def test_a_keep_root_item_stays_the_root():
    uf = UnionFind(keep_root=lambda x: x == "ground")
    uf.union("p", "q")
    uf.union("q", "ground")
    assert uf.find("p") == uf.find("q") == "ground"
    uf.union("ground", "r")
    assert uf.find("r") == "ground"


def test_groups_follow_insertion_order_not_root_identity():
    uf = UnionFind()
    for a, b in [("a", "b"), ("c", "d"), ("e", "a")]:
        uf.union(a, b)
    groups = uf.groups()
    assert [sorted(m) for m in groups.values()] == [["a", "b", "e"], ["c", "d"]]
