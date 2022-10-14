"""This file initializes the app and registers the blueprints."""


from flask import Flask


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(SECRET_KEY="dev")

    from . import auth, completion, model, database

    if test_config is not None:
        # load the test config if passed in
        app.config.from_mapping(test_config)
        database.init_testing_database()
    else:
        database.init_database()

    app.register_blueprint(auth.bp)
    app.register_blueprint(completion.bp)
    app.register_blueprint(model.bp)
    app.add_url_rule('/', endpoint='index')
    return app
