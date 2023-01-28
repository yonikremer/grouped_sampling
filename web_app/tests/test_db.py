"""This file contains all the tests for the database."""
from sqlite3 import Connection

from flask import Flask
from pytest import mark

from web_app.flaskr.database import get_db


def test_and_get_close_db(app: Flask):
    """Test that the close my_db is called when app is terminated."""
    with app.app_context():
        my_db: Connection = get_db()
        assert my_db is get_db()  # Assert that get_db()
        # returns a pointer to the same database every time


@mark.parametrize(
    "table_name",
    ("user", "model", "completion"),
)
def test_tables_are_not_empty(app: Flask, table_name: str):
    """Test that the tables of the database are not empty."""
    with app.app_context():
        my_db: Connection = get_db()
        my_cursor = my_db.cursor()
        my_cursor.execute(f"SELECT * FROM {table_name}")
        assert my_cursor.fetchone()
