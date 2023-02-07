"""Contains functions that configure the tests"""
from typing import Generator

import pytest
from flask import Flask
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from web_app.flaskr import create_app


@pytest.fixture
def app() -> Generator[Flask, None, None]:
    """Create and configure a new app instance for each test."""
    # create the app with common test config
    # Return app but don't exit the function.
    yield create_app({"TESTING": True})


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def runner(app: Flask):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()


class AuthActions:
    """Contains helper methods for testing the authentication"""

    def __init__(self, client: FlaskClient):
        self._client: FlaskClient = client

    def login(self, username: str = "test", password: str = "test") -> TestResponse:
        """Login helper function."""
        return self._client.post(
            "/auth/login", data={"username": username, "password": password}
        )

    def logout(self) -> TestResponse:
        """Logout helper function."""
        return self._client.get("/auth/logout")


@pytest.fixture
def auth(client: FlaskClient) -> AuthActions:
    """tests login and logout for an existing user"""
    return AuthActions(client)
