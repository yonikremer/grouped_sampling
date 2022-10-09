"""This file contains all the tests for the database."""
import sqlite3
from sqlite3 import Connection

import pytest
from web_app.flaskr.database import get_db
from flask import Flask


def test_get_close_db(app: Flask):
    """Test that the close my_db is called when app is terminated."""
    with app.app_context():
        my_db: Connection = get_db()
        assert my_db is get_db()  # Assert that get_db()
        # returns a pointer to the same database evert time

    with pytest.raises(sqlite3.ProgrammingError) as error:
        my_db.execute('SELECT 1')

    assert 'closed' in str(error.value)
