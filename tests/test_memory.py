from om1.memory import Memory

def test_memory_add_recall():
    m = Memory()
    m.add("key", "value")
    assert m.recall("key") == "value"
