"""Tests the completion blueprint"""
from collections.abc import Generator
from sqlite3 import Connection
from typing import Dict

from flask import Flask
from flask.testing import FlaskClient
from pytest import mark

from web_app.flaskr.database import get_db


def test_index(client: FlaskClient, auth) -> None:
    """Tests completion.html contains everything it supposed to"""
    response = client.get("/")
    assert response.status_code == 200
    lower_html_str = response.get_data(as_text=True).lower()
    assert "log in" in lower_html_str or "login" in lower_html_str
    assert "register" in lower_html_str
    assert not ("log out" in lower_html_str or "logout" in lower_html_str)

    auth.login()
    response = client.get("/")
    assert response.status_code == 200
    lower_html_str = response.get_data(as_text=True).lower()
    assert "log out" in lower_html_str or "logout" in lower_html_str
    assert not ("log in" in lower_html_str or "login" in lower_html_str)
    assert "register" not in lower_html_str


def count_completions(app: Flask) -> int:
    """Counts the number of completions in the database"""
    with app.app_context():
        my_db: Connection = get_db()
        return my_db.execute(
            "SELECT COUNT(id) as C FROM completion").fetchone()[0]


def valid_completion_create_requests() -> Generator[Dict[str, str]]:
    """Returns a pipeline of good completion/create requests"""
    fields = (
        "prompt",
        "num_tokens",
        "model_name",
        "group_size",
        "generation_type",
        "top_p",
        "top_k",
        "temperature",
    )
    request: Dict[str, str] = {f: "" for f in fields}
    # start with all default arguments
    yield request
    new_field_val_mapping = {
        "prompt": "This is a test prompt",
        "model_name": "gpt2",
        "group_size": "1",
        "num_tokens": "10",
        "generation_type": "sampling",
        "top_p": "0.9",
        "top_k": "10",
        "temperature": "0.9",
    }
    for field, val in new_field_val_mapping.items():
        request[field] = val
        yield request

    # now create sampling requests
    request["generation_type"] = "sampling"
    top_ps = ("0", "", "1", "0.2")
    top_ks = ("", "", "1", "10")
    request["top_p"] = ""
    for top_k in top_ks:
        request["top_k"] = top_k
        yield request
    request["top_k"] = ""
    for top_p in top_ps:
        request["top_p"] = top_p
        yield request
    yield request


def test_get_create(client: FlaskClient, app: Flask) -> None:
    """Tests the get request to the create page"""
    with client:
        response = client.get("/create")
        assert response.status_code == 200


@mark.parametrize(
    "curr_request",
    valid_completion_create_requests(),
)
def test_create(client: FlaskClient, auth, app: Flask,
                curr_request: Dict[str, str]) -> None:
    """tests you can create a new completion if you provide a valid request."""
    auth.login()
    client.post("/create", data=curr_request)
