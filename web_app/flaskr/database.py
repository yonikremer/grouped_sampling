"""This file contains the database functions."""
import os
import sqlite3

from flask import current_app, g


DATABASE_FILE_NAME = "flaskr.sqlite"
DATABASE_FOLDER = "instance"
DATABASE_FULL_PATH = os.path.join(DATABASE_FOLDER, DATABASE_FILE_NAME)


def get_db():
    """Opens a new database connection if there is none yet for the"""
    if "my_db" not in g:
        g.my_db = sqlite3.connect(DATABASE_FULL_PATH)
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
    with open("flaskr/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
