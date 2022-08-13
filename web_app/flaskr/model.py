"""Contains the functions and blueprint to upload a model and see a list of all models"""
import os

import tensorflow as tf
import keras
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, g
from werkzeug.utils import secure_filename
import pandas as pd

from flaskr.db import get_db
from flaskr.completion import complete
from flaskr.auth import login_required

bp = Blueprint('model', __name__)


@bp.route('/view_all', methods = ("POST", "GET"))
def view_all():
    """See the table models as an HTML page"""
    my_db = get_db()
    models = my_db.execute("SELECT * FROM model").fetchall()
    data_frame = pd.DataFrame(models)
    return render_template("model/view_all.html", table = [data_frame.to_html()], titles = data_frame.columns.values)


def file_is_allowed(filename):
    """return if the file has the correct extension"""
    banned_chars = {'/', ':', '*', '?', '<', '>', '|'}
    has_banned_chars = not any(char in filename for char in banned_chars)
    is_pb_file = filename.endswith(".pb")
    return is_pb_file and has_banned_chars


def model_errors(model_name):
    """Returns a list with all the errors when checking the model"""
    errors = []
    path = f"/flaskr/static/models/{model_name}.pb"
    model = tf.saved_model.load(path)
    if not isinstance(model, keras.Model):
        return ['The file is not a valid keras.Model']
    first_layer = model.get_layer(name = None, index = 0)
    if not isinstance(first_layer, keras.layers.Embedding):
        errors.append('The first layer of your model is not an Embedding layer')
    last_layer = model.get_layer(name = None, index = -1)
    if not isinstance(last_layer, keras.layers.Softmax):
        errors.append('The last layer of your model is not a Softmax layer')
    else:
        if last_layer.axis not in (-1, 2):
            errors.append('The last layer of your model is not a Softmax layer with axis=-1 or axis=2')
    test_completion = complete(model, "This is a test", 1.0, 1, 1, 1)
    if not isinstance(test_completion, str):
        errors.append('Your model does not generate the correct output. Please try a different model')
    return errors


def type_errors(my_request):
    """Returns all the type errors messages"""
    errors = ""
    arg_names = ("set_size", "num_layers", "num_heads", "d_model", "dff",
           "dropout_rate", "learning_rate", "batch_size")
    arg_types = (int, int, int, int, int, float, float, int)
    for _id, _type in zip(arg_names, arg_types):
        if _id not in my_request.form:
            errors += f'Missing {_id}'
        value = my_request.form[_id]
        if "e" in value.lower() or "**" in value or "^" in value:
            errors += 'e is not allowed in numbers, (use 0.01 and not 1e-2, 1E-2, 10^-2, 10**-2)'
        if str(_type(value)) != value:
            errors += f'{_id} is not a valid {_type}'
    return errors


@login_required
@bp.route('/upload', methods = ("POST", "GET"))
def upload():
    """A page where every user can upload their own model
    If uploading is successful, redirect to """
    if request.method == 'POST':
        # check if the post request has the file part
        errors = []
        if 'file' not in request.files:
            errors.append('No file part')
        file = request.files['file']
        if not file:  # if no file is selected => file is None
            errors.append('File is None')
        # if user does not select file, browser also
        # submit an empty part without model_name
        if file.filename == '':
            errors.append('Selected file does not have a name')
        if not file_is_allowed(file.filename):
            errors.append('File extension is not .pb')
        filename = secure_filename(file.filename)
        model_name = filename.split('.')[0]
        if len(errors) == 0:
            full_path = os.path.join(current_app.root_path, "static", "models", filename)
            file.save(full_path)
            flash('File successfully uploaded')
            errors = model_errors(model_name)  # The same as appending to errors because it's empty
            if len(errors) > 0:
                os.remove(full_path)
                errors.append('Model removed')

        set_size = int(request.form["set_size"])
        num_layers = int(request.form["num_layers"])
        num_heads = int(request.form["num_heads"])
        d_model = int(request.form["d_model"])
        dff = int(request.form["dff"])
        dropout_rate = float(request.form["dropout_rate"])
        learning_rate = float(request.form["learning_rate"])
        batch_size = int(request.form["batch_size"])
        errors += type_errors(request)
        if len(errors) == 0:
            my_db = get_db()
            my_db.execute("""INSERT INTO model
                          (user_id, model_name, set_size, num_layers, num_heads, 
                          d_model, dff, dropout_rate, learning_rate, batch_size)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (g.user["id"], model_name, set_size, num_layers, num_heads, d_model, 
                          dff, dropout_rate, learning_rate, batch_size))
            return redirect(url_for("model.view_all"))
        flash('\n'.join(errors))

    return render_template("model/upload.html")
