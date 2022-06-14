"""Contains the functions for the authentication."""

import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from flaskr.db import get_db

import sqlite3

from typing import Optional

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=('GET', 'POST'))
def register():
    """Register a new user."""
    if request.method == 'POST':
        username: str = request.form['username']
        password: str = request.form['password']
        db: sqlite3 = get_db()
        error: Optional[str] = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            try:
                db.execute(
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()  # commit the changes to the database
            except db.IntegrityError:
                error = "User {username} is already registered."
            else:
                return redirect(url_for("auth.login"))

        flash(error)  # flash is a function that is used to display messages to the user

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    """Logs in a user."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        user = db.execute(
            'SELECT * FROM user WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('index'))

        flash(error)

    return render_template('auth/login.html')


@bp.before_app_request
def load_logged_in_user():
    """If a user id is stored in the session, load the user object from the database into g.user."""
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()


@bp.route('/logout')
def logout():
    """clear session and logout"""
    session.clear()
    return redirect(url_for('index'))


def login_required(view):
    """View decorator that redirects anonymous users to the login page.
    args: view: function
    returns: function"""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        """the wrapped function"""
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view
