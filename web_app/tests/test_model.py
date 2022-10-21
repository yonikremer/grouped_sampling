"""Contains the tests for the model blueprint."""
from typing import List

from flask import Flask
from flask.testing import FlaskClient

from web_app.tests.conftest import AuthActions
from web_app.tests.test_base_html import test_base_html


def test_view_all(app: Flask, client: FlaskClient, auth: AuthActions) -> None:
    """Tests that the view_all function returns the correct data."""
    test_base_html(client, auth, "/view_all")

    components: List[str] = [
        "View All Models",
        "username",
        "created",
        "model_name",
        ]
    response = client.get("/view_all")

    for component in components:
        assert component in response.get_data(as_text=True)
