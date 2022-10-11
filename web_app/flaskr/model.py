"""Contains the functions and blueprint to upload a model and see a list of all models"""

from flask import Blueprint, render_template, g
import pandas as pd

from .auth import is_logged_in, UnAuthorizedException
from .database import get_db

bp = Blueprint('model', __name__)


@bp.route('/view_all', methods=("POST", "GET"))
def view_all():
    """See the table models as an HTML page"""
    my_db = get_db()
    models = my_db.execute("SELECT * FROM model").fetchall()
    data_frame = pd.DataFrame(models)
    return render_template("model/view_all.html", table=[data_frame.to_html()], titles=data_frame.columns.values)


def get_model_id(my_model_name: str) -> int:
    """Adds a model to the database if it doesn't exist already and returns the id of the model"""
    my_db = get_db()
    # Check if the model already exists
    model_exists: bool = my_db.execute(
        "SELECT EXISTS(SELECT 1 FROM model WHERE model_name=? LIMIT 1)",
        (my_model_name,))
    if not model_exists:
        if not is_logged_in():
            raise UnAuthorizedException("add a model to the database")
        my_db.execute(
            "INSERT INTO model (model_name, user_id) VALUES (?, ?)",
            (my_model_name, g.user['id'])
        )
    model_id: int = my_db.execute(
        "SELECT id FROM model WHERE model_name=?",
        (my_model_name,)
    ).fetchone()
    my_db.commit()
    return model_id
