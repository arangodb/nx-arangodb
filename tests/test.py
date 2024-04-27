import pytest

from .conftest import db

def test_db():
    assert db.version()