"""This file contains the database functions."""
import os
import sqlite3

from flask import g


DATABASE_FILE_NAME = "flaskr.sqlite"
DATABASE_FOLDER = "instance"
DATABASE_FULL_PATH = os.path.join(DATABASE_FOLDER, DATABASE_FILE_NAME)
SQL_SCHEMA_SCRIPT_PATH = "flaskr/schema.sql"
TESTING_DATABASE_PATH = os.path.join(os.path.dirname(__file__), "testing_data.sqlite")
TESTING_DATA_PATH = os.path.join(os.path.dirname(__file__), "tests", "testing_data.sql")


def get_db(testing: bool = False) -> sqlite3.Connection:
    """Opens a new database connection if there is none yet for the"""
    wanted_db_path = TESTING_DATABASE_PATH if testing else DATABASE_FULL_PATH
    if "my_db" not in g:
        g.my_db = sqlite3.connect(wanted_db_path)
        g.my_db.row_factory = sqlite3.Row

    return g.my_db


# noinspection PyUnusedLocal
def close_db(e=None):
    """Closes the database again at the end of the request.
    sometimes this is called with an argument e, which is ignored."""
    my_db = g.pop("my_db", None)

    if my_db is not None:
        my_db.close()


def init_database() -> None:
    """Deletes and recreates the database."""

    if not os.path.exists(DATABASE_FOLDER):
        os.makedirs(DATABASE_FOLDER)

    conn = sqlite3.connect(DATABASE_FULL_PATH)
    with open(SQL_SCHEMA_SCRIPT_PATH) as f:
        conn.executescript(f.read())
    conn.close()


def init_testing_database():
    """Initializes the database for testing."""

    if not os.path.exists(DATABASE_FOLDER):
        os.makedirs(DATABASE_FOLDER)

    conn = sqlite3.connect(TESTING_DATABASE_PATH)
    with open(SQL_SCHEMA_SCRIPT_PATH) as f:
        conn.executescript(f.read())
    with open(TESTING_DATA_PATH) as f:
        conn.executescript(f.read())
    conn.close()
