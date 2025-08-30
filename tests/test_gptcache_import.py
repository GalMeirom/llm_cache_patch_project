def test_gptcache_import():
    import importlib

    m = importlib.import_module("gptcache")
    assert hasattr(m, "__file__")
