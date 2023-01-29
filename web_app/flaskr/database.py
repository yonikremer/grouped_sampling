"""This file contains the database functions."""
import os
import sqlite3

from flask import current_app, g


def parent_directory(original_directory: str) -> str:
    """Returns the parent directory of the original directory."""
    return os.path.abspath(os.path.join(original_directory, os.pardir))


this_files_dir = os.path.dirname(__file__)
# C:\Users\yonik\grouped_sampling\web_app\flaskr
this_dir_parent = parent_directory(this_files_dir)
# C:\Users\yonik\grouped_sampling\web_app

DATABASE_FOLDER = os.path.join(this_dir_parent, "instance")
# C:\Users\yonik\grouped_sampling\web_app\instance
DATABASE_FULL_PATH = os.path.join(DATABASE_FOLDER, "flaskr.sqlite")
# C:\Users\yonik\grouped_sampling\web_app\instance\flaskr.sqlite
TESTING_DATABASE_PATH = os.path.join(DATABASE_FOLDER, "testing_data.sqlite")
# C:\Users\yonik\grouped_sampling\web_app\flaskr\testing_data.sqlite
SQL_SCHEMA_SCRIPT_PATH = os.path.join(this_files_dir, "schema.sql")
# C:\Users\yonik\grouped_sampling\web_app\flaskr\schema.sql
TESTING_DATA_PATH = os.path.join(this_dir_parent, "tests", "testing_data.sql")
# C:\Users\yonik\grouped_sampling\web_app\tests\testing_data.sql


def get_db() -> sqlite3.Connection:
    """Opens a new database connection if there is none yet for the"""
    testing: bool = current_app.config["TESTING"]
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
