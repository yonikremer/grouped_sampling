"""configures the tests"""
from web_app.flaskr.__init__ import create_app


def test_config() -> None:
    """Test create_app without passing test config."""
    assert not create_app().testing
    assert create_app({'TESTING': True}).testing
