"""Tests for download_demos.py"""

from csgoml.scripts.download_demos import find_missing


class TestDownloadDemos:
    """Class to test download_demos.py"""

    def test_find_missing(self):
        """Tests find_missing"""
        assert find_missing({1, 2, 5, 6, 7, 9}) == [3, 4, 8]
        assert find_missing({1, 2, 3, 4, 5}) == []
        assert find_missing(set()) == []
        assert find_missing({6}) == []
