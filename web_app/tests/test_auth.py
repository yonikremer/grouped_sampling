"""Test the auth blueprint"""

from pytest import mark
from flask import g, session, Flask, Response
from flask.testing import FlaskClient

from web_app.flaskr.database import get_db
from web_app.tests.conftest import AuthActions


def test_register(client: FlaskClient, app: Flask) -> None:
    # test that viewing the page renders without template errors
    assert client.get("/auth/register").status_code == 200

    # test that successful registration redirects to the login page
    response = client.post("/auth/register", data={"username": "a", "password": "a"})
    assert response.headers["Location"] == "/"

    # test that the user was inserted into the database
    with app.app_context():
        my_db = get_db()
        row = my_db.execute("SELECT * FROM user WHERE username = 'a'").fetchone()
        assert row is not None


@mark.parametrize(
    ("username", "password", "message"),
    (
        ("", "", b'Username is required.'),
        ("a", "", b'Password is required.'),
        ("test", "test", b'already registered'),
    ),
)
def test_register_validate_input(client: FlaskClient, username: str, password: str, message: bytes) -> None:
    """
    Tests that you:
    1 can't create a user with existing username
    2 can't create a user with no password
    3 can't register twice
    """
    response = client.post(
        "/auth/register", data={"username": username, "password": password}
    )
    assert message in response.data


def test_login(client: FlaskClient, auth: AuthActions):
    """Tests you can login"""
    # test that viewing the page renders without template errors
    assert client.get("/auth/login").status_code == 200

    # test that successful login redirects to the index page
    response: Response = auth.login()
    assert response.headers["Location"] == "/"

    # login request set the user_id in the session
    # check that the user is loaded from the session
    with client:
        client.get("/")
        assert session["user_id"] == 1
        assert g.user["username"] == "test"


@mark.parametrize(
    ("username", "password", "message"),
    (("a", "test", 'Incorrect username'), ("test", "a", 'Incorrect password')),
)
def test_login_validate_input(auth: AuthActions, username: str, password: str, message: bytes) -> None:
    """Tests you can't log in to none existing users or with wrong password"""
    response: Response = auth.login(username, password)
    assert message in response.get_data(as_text=True)


def test_logout(client: FlaskClient, auth: AuthActions):
    """Tests the logout feature"""
    auth.login()

    with client:
        auth.logout()
        assert "user_id" not in session


@mark.parametrize(
    ("valid_username", "valid_password"),
    (("a", "a"), ("this_is_a_test", "this_is_also_a_test")),
)
def test_register_logout_login(client: FlaskClient, auth: AuthActions, valid_username, valid_password):
    """Tests that you can register, logout and login again with the same username and password."""
    assert client.get("/").status_code == 200
    client.post("/auth/register", data={"username": valid_username, "password": valid_password})
    assert client.get("/").status_code == 200
    auth.logout()
    assert client.get("/").status_code == 200
    client.post("/auth/login", data={"username": valid_username, "password": valid_password})
    assert client.get("/").status_code == 200
    with client:
        client.get("/")
        assert g.user["username"] == valid_username

    # logout
    auth.logout()
