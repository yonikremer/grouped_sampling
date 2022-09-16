"""Contains the functions and blueprint to upload a model and see a list of all models"""
import os

from flask import Blueprint, render_template, g
import pandas as pd

from flaskr.db import get_db
from flaskr.auth import login_required

bp = Blueprint('model', __name__)


@bp.route('/view_all', methods = ("POST", "GET"))
def view_all():
    """See the table models as an HTML page"""
    my_db = get_db()
    models = my_db.execute("SELECT * FROM model").fetchall()
    data_frame = pd.DataFrame(models)
    return render_template("model/view_all.html", table = [data_frame.to_html()], titles = data_frame.columns.values)


@bp.route('/new', methods = ("Post", "GET"))
@login_required
def new(model_name):
    """Upload a new model"""
    my_db = get_db()
    my_db.execute("INSERT INTO model (name, user_id) VALUES (?, ?)", (model_name, g.user['id']))
