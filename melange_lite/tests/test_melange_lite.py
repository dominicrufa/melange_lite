"""
Unit and regression test for the melange_lite package.
"""

# Import package, test suite, and other packages as needed
import melange_lite
import pytest
import sys

def test_melange_lite_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "melange_lite" in sys.modules
