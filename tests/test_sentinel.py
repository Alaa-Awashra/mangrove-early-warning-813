import pytest

from src.sentinel import boa_add_offset


def test_processing_baseline_03_has_no_offset():
    assert boa_add_offset("03.00") == 0


def test_processing_baseline_04_and_05_use_negative_1000_offset():
    assert boa_add_offset("04.00") == -1000
    assert boa_add_offset("05.11") == -1000


def test_unverified_processing_baseline_fails_closed():
    with pytest.raises(ValueError):
        boa_add_offset("06.00")
