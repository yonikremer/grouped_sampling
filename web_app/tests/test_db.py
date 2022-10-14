"""This file contains all the tests for the database."""
from sqlite3 import Connection

from web_app.flaskr.database import get_db
from flask import Flask


def test_and_get_close_db(app: Flask):
    """Test that the close my_db is called when app is terminated."""
    with app.app_context():
        my_db: Connection = get_db(testing=True)
        assert my_db is get_db(testing=True)  # Assert that get_db()
        # returns a pointer to the same database every time
