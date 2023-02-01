"""Contains the functions for the authentication."""

import functools

from flask import (Blueprint, flash, g, redirect, render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from .database import get_db

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=('GET', 'POST'))
def register():
    """
    Register a new user.
    If the registration is successful, the user is directed to auth/login
    else, the user is directed to auth/register recursively until a successful registration
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        my_db = get_db()
        error = None

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
            else:  # if no error
                session.clear()
                session['user_id'] = my_db.execute(
                    "SELECT id FROM user WHERE username = ?", (username,)
                ).fetchone()['id']
                return redirect(url_for('index'))

        flash(error)  # flash is a function that is used to display messages to the user

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    """
    Logs in a user.
    If the logging is successful, the user is directed to the main page
    else, the user is directed to auth/login recursively until a successful registration
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        my_db = get_db()
        error = None
        user = my_db.execute(
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
def load_logged_in_user():
    """If a user id is stored in the session, load the user object from the database into g.user."""
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
                "SELECT * FROM user WHERE id = ?", (user_id,)
        ).fetchone()


@bp.route('/logout')
def logout():
    """clears session, logs out and redirects to the main page"""
    session.clear()
    return redirect(url_for('index'))


def login_required(view):
    """
    View decorator that redirects anonymous users to the login page.
    args: view: function
    returns: function
    """
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        """the wrapped function"""
        if not is_logged_in():
            flash('Logging in is required for this page.')
            return redirect(url_for('auth.login'))

        return view(*args, **kwargs)

    return wrapped_view


def is_logged_in() -> bool:
    """
    Checks if the user is logged in
    returns: bool
    """
    return g.user is not None
