"""Contains the tests for the model blueprint."""
from typing import List

from pytest import mark
from flask import Flask
from flask.testing import FlaskClient

from web_app.tests.conftest import AuthActions
from web_app.tests.test_base_html import test_base_html
from web_app.flaskr.model import get_model_id


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


@mark.parametrize("model_name_and_index", [("test_model1", 1), ("test_model2", 2), ("test_model3", 3)])
def test_get_model_id_existing(app: Flask, model_name_and_index: (str, int)) -> None:
    """Tests that get_model_id works for existing model names"""
    model_name, model_index = model_name_and_index
    with app.app_context():
        assert get_model_id(model_name) == model_index

# get model id for non-existing model names and add_model_to_db are tested when testing completion/create
