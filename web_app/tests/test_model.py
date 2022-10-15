"""Contains the tests for the model blueprint."""
from typing import List
from flask.testing import FlaskClient

from web_app.tests.conftest import AuthActions


def test_view_all(client: FlaskClient, auth: AuthActions) -> None:
    """Tests that the view_all function returns the correct data."""
    response = client.get('model/view_all')
    assert response.status_code == 200
    # Assert that the view_all.html extends from base.html.
    assert b'Log In' in response.data
    assert b'Register' in response.data
    assert b'Log Out' not in response.data

    auth.login()
    assert b'Log In' not in response.data
    assert b'Register' not in response.data

    components: List[str] = [
        "View All Models",
        "Upload your own models!",
        "user_id",
        "created",
        "model_name",
        "set_size",
        "num_layers",
        "d_model",
        "dff",
        "num_heads",
        "dropout_rate",
        "learning_rate",
        "batch_size",
        "id",
        "128",
        "64",
        "0.1",
        'Log Out']

    for component in components:
        assert bytes(component) in response.data


def test_model_errors(client: FlaskClient, auth: AuthActions) -> None:
    """Tests the model errors function"""
    raise NotImplementedError


def test_upload():
    """Tests the uploading function"""
    raise NotImplementedError
