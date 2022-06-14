"""This file contains all the tests for the database."""
import sqlite3

import pytest
from flaskr.db import get_db
from flask import Flask


def test_get_close_db(app: Flask):
    """Test that the close db is called when app is terminated."""
    with app.app_context():
        db = get_db()
        assert db is get_db()  # Assert that get_db()
        # returns a pointer to the same database evert time

    with pytest.raises(sqlite3.ProgrammingError) as error:
        db.execute('SELECT 1')

    assert 'closed' in str(error.value)


def test_init_db_command(runner, monkeypatch):
    """Test the init-db command."""
    class Recorder:
        """Records events."""
        called = False

    def fake_init_db():
        """sets the Recorder.called to True."""
        Recorder.called = True

    monkeypatch.setattr('flaskr.db.init_db', fake_init_db)
    result = runner.invoke(args=['init-db'])
    assert 'Initialized' in result.output
    assert Recorder.called
