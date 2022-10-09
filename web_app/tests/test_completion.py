"""Tests the completion blueprint"""

from sqlite3 import Connection
from typing import List

from flask import Flask
from flask.testing import FlaskClient

from web_app.flaskr.database import get_db


def test_index(client: FlaskClient, auth) -> None:
    """Tests completion.html contains everything it supposed to"""
    response = client.get('/')
    assert (b'Log In' in response.data)
    assert (b'Register' in response.data)
    assert not (b'Log Out' in response.data)

    auth.login()
    response = client.get('/')
    components: List[bytes] = [b'Log Out', b'test title', b'by test on 01/01/18 00:00',
                               b'Is this a test?', b'Of course it is just a test!', b'Try It yourself!']
    for com in components:
        assert (com in response.data)
    assert not (b'Log In' in response.data)
    assert not (b'Register' in response.data)


def test_create(client: FlaskClient, auth, app: Flask) -> None:
    """tests you can create a new completion"""
    auth.login()
    assert (client.get('/create').status_code == 200)
    client.post('/create', data={'prompt': 'test prompt', 'model_id': 1})
    with app.app_context():
        my_db: Connection = get_db()
        count = my_db.execute('SELECT COUNT(id) FROM completion').fetchone()[0]
        assert (count == 3)
        # there are 3 completions in the database, two from testing_data.sql and one from the test
