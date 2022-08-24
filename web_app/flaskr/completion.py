"""Contains the functions for the completion page and the completion blueprint"""
import os

import tensorflow as tf
from flask import Blueprint, g, redirect, render_template, request, url_for, flash
from transformers import AutoTokenizer, AutoModelForCausalLM

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import init_tf_tokenizer
from flaskr.sampling import complete


bp = Blueprint('completion', __name__)


@bp.route('/')
def index():
    """The page with all the completions"""
    my_db = get_db()
    # Execute a SQL query and return the results
    query = "SELECT * FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC"
    completions = my_db.execute(query).fetchall()
    return render_template("completion/index.html", completions = completions)


def add_answer_to_db(model_name, prompt, answer):
    """Adds an answer to the database"""
    connection = get_db()
    # get model id
    model_id_query = f"SELECT id FROM models WHERE name = ?"
    model_id = connection.execute(model_id_query, (model_name)).fetchone()
    query_structure = "INSERT INTO completion (model_id, prompt, answer, user_id) VALUES (?, ?, ?, ?)"
    connection.execute(query_structure, (model_id, prompt, answer, g.user['id']))
    connection.commit()


@login_required
@bp.route('/create', methods = ('GET', 'POST'))
def create():
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == "POST":
        print("called function create")
        top_p_str = request.form['top_p']
        top_k_str = request.form['top_k']
        num_groups = request.form['num_groups']
        prompt = request.form['prompt']
        model_name = request.form['model_name']
        group_size = request.form['group_size']
        num_groups_int = int(num_groups)
        top_p_float = float(top_p_str)
        top_k = int(top_k_str)
        pb_model_path = f"/flaskr/static/models/{model_name}.pb"
        h5_model_path = f"/flaskr/static/models/{model_name}.h5"

        if not model_name in g.loaded_models.keys():
            if os.path.exists(pb_model_path):
                g.loaded_models[model_name] = tf.saved_model.load(pb_model_path)
                g.loaded_tokenizers[model_name], _ = init_tf_tokenizer()

            elif os.path.exists(h5_model_path):
                g.loaded_models[model_name] = tf.keras.models.load_model(h5_model_path)
                g.loaded_tokenizers[model_name], _ = init_tf_tokenizer()

            else:
                g.loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(model_name)
                g.loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        print("loaded model and tokenizer")
        completions = complete(model_name, prompt, top_p_float, top_k, num_groups_int, group_size)
        try:
            best_completion = max(completions, key=completions.get)
        except ValueError:
            flash("No completions were found, try using higher top-k or top-p or smaller group size")
            flash("That happend because all the models had repeated tokens")
            return redirect(url_for('completion/index.html'))
        add_answer_to_db(model_name, prompt, best_completion)
        return redirect(url_for('completion/index.html'))
    return render_template("completion/create.html")
