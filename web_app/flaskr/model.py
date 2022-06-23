"""Contains the functions and blueprint to upload a model and see a list of all models"""
import os
from typing import List, Union, Set
import tensorflow as tf

from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, g, Response
from sqlite3 import Connection

from werkzeug import Response
from werkzeug.utils import secure_filename

from flaskr.db import get_db
from flaskr.completion import complete
import pandas as pd
from flaskr.auth import login_required

bp = Blueprint('model', __name__)


@bp.route('/view_all', methods = ("POST", "GET"))
def view_all() -> str:
    """See the table models as an HTML page"""
    my_db: Connection = get_db()
    models: List[dict] = my_db.execute("SELECT * FROM model").fetchall()
    df: pd.DataFrame = pd.DataFrame(models)
    return render_template("model/view_all.html", table = [df.to_html()], titles = df.columns.values)


def allowed_file(filename: str) -> bool:
    """return if the file has the correct extension"""
    banned_chars: Set[chr] = {'/', ':', '*', '?', '<', '>', '|'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == "pb" and not any(char in filename for char in banned_chars)


def model_errors(model_name: str) -> List[str]:
    """Returns a list with all the errors when checking the model"""
    errors: List[str] = []
    path: str = f"/flaskr/static/models/{model_name}.pb"
    model: tf.keras.Model = tf.saved_model.load(path)
    if not isinstance(model, tf.keras.Model):
        return ['The file is not a valid tf.keras.Model']
    first_layer: tf.keras.layers.Layer = model.get_layer(name = None, index = 0)
    if not isinstance(first_layer, tf.keras.layers.Embedding):
        errors.append('The first layer of your model is not an Embedding layer')
    last_layer: tf.keras.layers.Layer = model.get_layer(name = None, index = -1)
    if not isinstance(last_layer, tf.keras.layers.Softmax):
        errors.append('The last layer of your model is not a Softmax layer')
    else:
        if last_layer.axis not in (-1, 2):
            errors.append('The last layer of your model is not a Softmax layer with axis=-1 or axis=2')
    test_completion: Union[Response, str] = complete("This is a test", model_name)
    if not isinstance(test_completion, str):
        errors.append('Your model does not generate the correct output. Please try a different model')
    return errors


def type_errors(my_request: request) -> List[str]:
    """Returns all the type errors messages"""
    errors = []
    ids: List[str] = ["set_size", "num_layers", "num_heads", "d_model", "dff", "dropout_rate", "learning_rate",
                      "batch_size"]
    types: List[type] = [int, int, int, int, int, float, float, int]
    for _id, _type in zip(ids, types):
        if _id not in my_request.form:
            errors.append(f'Missing {_id}')
        value: str = my_request.form[_id]
        if "e" in value.lower() or "**" in value or "^" in value:
            errors.append('e is not allowed in numbers, (use 0.01 and not 1e-2, 1E-2, 10^-2, 10**-2)')
        if str(_type(value)) != value:
            errors.append(f'{_id} is not a valid {_type}')
    return errors


@login_required
@bp.route('/upload', methods = ("POST", "GET"))
def upload() -> Union[Response, str]:
    """A page where every user can upload their own model
    If uploading is successful, redirect to """
    if request.method == 'POST':
        # check if the post request has the file part
        errors: List[str] = []
        if 'file' not in request.files:
            errors.append('No file part')
        file = request.files['file']
        if not file:  # if no file is selected => file is None
            errors.append('File is None')
        # if user does not select file, browser also
        # submit an empty part without model_name
        if file.filename == '':
            errors.append('Selected file does not have a name')
        if not allowed_file(file.filename):
            errors.append('File extension is not .pb')
        filename: str = secure_filename(file.filename)
        model_name: str = filename.split('.')[0]
        if len(errors) == 0:
            full_path: str = os.path.join(current_app.root_path, "static", "models", filename)
            file.save(full_path)
            flash('File successfully uploaded')
            errors = model_errors(model_name)  # The same as appending to errors because it's empty
            if len(errors) > 0:
                os.remove(full_path)
                errors.append('Model removed')

        set_size: int = int(request.form["set_size"])
        num_layers: int = int(request.form["num_layers"])
        num_heads: int = int(request.form["num_heads"])
        d_model: int = int(request.form["d_model"])
        dff: int = int(request.form["dff"])
        dropout_rate: float = float(request.form["dropout_rate"])
        learning_rate: float = float(request.form["learning_rate"])
        batch_size: int = int(request.form["batch_size"])
        errors += type_errors(request)
        if len(errors) == 0:
            my_db: Connection = get_db()
            my_db.execute("INSERT INTO model "
                          "(user_id, model_name, set_size, num_layers, num_heads, d_model, dff, dropout_rate, learning_rate, batch_size)"
                          " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (g.user["id"], model_name, set_size, num_layers, num_heads, d_model, dff, dropout_rate,
                           learning_rate, batch_size))
            return redirect(url_for("model.view_all"))
        flash('\n'.join(errors))

    return render_template("model/upload.html")
