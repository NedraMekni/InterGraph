"""
Unit and regression test for the InterGraph package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import InterGraph


def test_InterGraph_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "InterGraph" in sys.modules
