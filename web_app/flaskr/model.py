"""Contains the functions and blueprint to upload a model and see a list of all models"""
from sqlite3 import Connection, Row

from flask import Blueprint, render_template, g
import pandas as pd

from .auth import login_required
from .database import get_db

bp = Blueprint('model', __name__)


@bp.route('/view_all', methods=("POST", "GET"))
def view_all():
    """See the table models as an HTML page"""
    # TODO: fix this method so it does something
    my_db = get_db()
    models = my_db.execute("SELECT * FROM model").fetchall()
    data_frame = pd.DataFrame(models)
    return render_template("model/view_all.html", table=[data_frame.to_html()], titles=data_frame.columns.values)


@login_required
def add_model_to_db(my_model_name: str, my_db: Connection) -> None:
    """Adds a new model to the database, assuming the model doesn't already exist"""
    my_db.execute(
        "INSERT INTO model (model_name, user_id) VALUES (?, ?)",
        (my_model_name, g.user['id'])
    )
    my_db.commit()


def get_model_id(my_model_name: str) -> int:
    """Adds a model to the database if it doesn't exist already and returns the id of the model"""
    my_db = get_db()
    # Check if the model already exists
    model_row: Row = my_db.execute(
        "SELECT id FROM model WHERE model_name=?",
        (my_model_name,)
    ).fetchone()
    model_id = model_row['id'] if model_row else None
    my_db.commit()
    if model_id is None:
        add_model_to_db(my_model_name, my_db)
        model_row: Row = my_db.execute(
            "SELECT id FROM model WHERE model_name=?",
            (my_model_name,)
        ).fetchone()
        model_id = model_row['id'] if model_row else None
    if not isinstance(model_id, int):
        raise TypeError("model_id is not an int")
    return model_id
