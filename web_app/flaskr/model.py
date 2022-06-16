"""Contains the functions and blueprint to upload a model and see a list of all models"""
import os
from typing import List

from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from sqlite3 import Connection

from werkzeug.utils import secure_filename

from db import get_db
import pandas as pd
from auth import login_required

bp = Blueprint('completion', __name__)


@bp.route('/view_all', methods = ("POST", "GET"))
def view_all():
    """See the table models as an HTML page"""
    db: Connection = get_db()
    models: List[dict] = db.execute("SELECT * FROM model").fetchall()
    df: pd.DataFrame = pd.DataFrame(models)
    return render_template("model/view_all.html", table = [df.to_html()], titles = df.columns.values)


def allowed_file(filename: str) -> bool:
    """return if the file has the correct extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == "pb"


@login_required
@bp.route('/upload', methods = ("POST", "GET"))
def upload():
    """A page where every user can upload their own model
    If uploading is successful, redirect to """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template("model/upload.html")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template("model/upload.html")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('view_all'))
