"""This file contains the database functions."""

import sqlite3

from flask import current_app, g


def get_db():
    """Opens a new database connection if there is none yet for the"""
    if "my_db" not in g:
        g.my_db = sqlite3.connect(
            current_app.config["DATABASE"],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.my_db.row_factory = sqlite3.Row

    return g.my_db


# noinspection PyUnusedLocal
def close_db(e=None):
    """Closes the database again at the end of the request.
    sometimes this is called with an argument e, which is ignored."""
    my_db = g.pop("my_db", None)

    if my_db is not None:
        my_db.close()
