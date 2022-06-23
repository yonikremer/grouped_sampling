"""Contains the functions for the authentication."""

import functools
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, Response
)
from werkzeug.security import check_password_hash, generate_password_hash
from flaskr.db import get_db
from sqlite3 import Connection
from typing import Optional, Union

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=('GET', 'POST'))
def register():
    """Register a new user.
    If the registration is successful, the user is directed to auth.login
    else, the user is directed to auth.register recursively until a successful registration"""
    if request.method == 'POST':
        username: str = request.form['username']
        password: str = request.form['password']
        my_db: Connection = get_db()
        error: Optional[str] = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            try:
                my_db.execute(
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                my_db.commit()  # commit the changes to the database
            except my_db.IntegrityError:
                error = f"User {username} is already registered."
            else:
                return redirect(url_for("auth.login"))

        flash(error)  # flash is a function that is used to display messages to the user

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login() -> Union[Response, str]:
    """Logs in a user.
    If the loging is successful, the user is directed to the main page
    else, the user is directed to auth.login recursively until a successful registration"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        my_db: Connection = get_db()
        error = None
        user: dict = my_db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
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
def load_logged_in_user() -> None:
    """If a user id is stored in the session, load the user object from the database into g.user."""
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
                "SELECT * FROM user WHERE id = ?", (user_id,)
        ).fetchone()


@bp.route('/logout')
def logout() -> Response:
    """clears session, logs out and redirects to the main page"""
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
            flash('Loging in is required for this page.')
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view
