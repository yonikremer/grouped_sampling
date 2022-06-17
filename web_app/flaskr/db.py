"""This file contains the database functions."""

import sqlite3
from sqlite3 import Connection

import click
from flask import current_app, g, Flask
from flask.cli import with_appcontext


def get_db() -> Connection:
    """Opens a new database connection if there is none yet for the"""
    if "my_db" not in g:
        g.my_db = sqlite3.connect(
            current_app.config["DATABASE"],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.my_db.row_factory = sqlite3.Row

    return g.my_db


# noinspection PyUnusedLocal
def close_db(e=None) -> None:
    """Closes the database again at the end of the request.
    sometimes this is called with an argument e, which is ignored."""
    my_db: Connection = g.pop("my_db", None)

    if my_db is not None:
        my_db.close()


def init_db() -> None:
    """Executes the SQL commands in my_db.sql to delete and recreate the database."""
    my_db: Connection = get_db()
    with current_app.open_resource('schema.sql') as f:
        my_db.executescript(f.read().decode('utf8'))


@with_appcontext
@click.command('init-db')  # define command "init-db"
def init_db_command() -> None:
    """Calls init_db() when the command 'init-bd' is called."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app: Flask) -> None:
    """Register database functions with the Flask app. This is called by the application factory."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
