"""This file initializes the app and registers the blueprints."""

import os

from flask import Flask

DATABASE_FILE_NAME = "flaskr.sqlite"
DATABASE_FOLDER = "instance"


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app_folder: str = os.path.dirname(os.path.abspath(__file__))

    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(f"{app_folder}/{DATABASE_FOLDER}", DATABASE_FILE_NAME),
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    # ensure the instance folder exists
    if not os.path.exists(app.instance_path):
        os.makedirs(app.instance_path)

    from . import auth, completion, model
    app.register_blueprint(auth.bp)
    app.register_blueprint(completion.bp)
    app.register_blueprint(model.bp)
    app.add_url_rule('/', endpoint='index')
    return app
