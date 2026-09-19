import opensysone


def test_version_is_a_string():
    assert isinstance(opensysone.__version__, str)
    assert opensysone.__version__.count(".") >= 2
