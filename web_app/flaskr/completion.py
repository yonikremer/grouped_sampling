"""Contains the functions for the completion page"""
from typing import Optional, List, Union
import tensorflow as tf
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from werkzeug import Response

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import tokenize, init_tokenizer
from werkzeug.exceptions import abort
import os
from sqlite3 import Connection

bp = Blueprint('completion', __name__)


@bp.route('/')
def index():
    """The page with all the completions"""
    db: Connection = get_db()
    # Execute a SQL query and return the results
    completions: List[dict] = db.execute(
        'SELECT c.id, user_id, model_id, created, prompt, answer FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC').fetchall()
    return render_template('completion/index.html', completions = completions)


def complete(prompt: str, model_name: str) -> Union[Response, str]:
    """Loads the model, preprocess the prompt and completes the text"""
    path: str = f"/flaskr/static/models/{model_name}"
    if not os.path.exists(path):
        raise ValueError(f"Model {model_name} does not exist or have a wrong name")
    model: tf.keras.model.Model = tf.saved_model.load(path)
    tokenizer_prompt: tf.Tensor = tokenize(prompt)
    if tokenizer_prompt.shape[0] > g.MAX_LEN:
        flash('The prompt is too long')
        return redirect(url_for('completion.index'))
    tokenized_ans = model.predict(tokenizer_prompt)
    string_tensor_ans: tf.RaggedTensor = g.tokenizer.detokenize(tokenized_ans)
    string_words: List[str] = []
    for one_d_tensor in string_tensor_ans:
        zero_d_tensor: tf.Tensor
        for zero_d_tensor in one_d_tensor:
            encoded = str(zero_d_tensor.numpy())  # "b'the string'"
            string_word = encoded[2:-1]  # "the string"
            string_words.append(string_word)

    sentence: str = ' '.join(string_words)
    remove_words = {'[PAD]', '[UNK]', '[START]', '[END]', '[MASK]'}
    for word in remove_words:
        sentence = sentence.replace(word, '')
    return sentence


@login_required
@bp.route('/create', methods = ('GET', 'POST'))
def create():
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == 'POST':

        if not hasattr(g, 'my_tokenizer'):
            my_tokenizer = init_tokenizer()
            g.my_tokenizer = my_tokenizer
        prompt: str = request.form['prompt']
        model_id: int = int(request.form['model_id'])
        error: Optional[str] = None
        if not prompt:
            error = 'Title is required.'
        if not model_id:
            error = 'You must select a model.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            model_name: str = db.execute('SELECT model_name FROM model WHERE id = ?', (model_id,)).fetchone()
            if model_name is None:
                flash('Model does not exist')
                return redirect(url_for('completion.create'))
            answer: str = complete(prompt, model_name)
            db.execute(
                'INSERT INTO completion (model_id, prompt, answer, author_id)'
                ' VALUES (?, ?, ?, ?)',
                (model_id, prompt, answer, g.user['id'])
            )
            db.commit()
            return redirect(url_for('completion/index.html'), )
    return render_template("completion/create.html")


def get_completion(com_id: int) -> dict:
    """Get a completion with id com_id from the database"""
    completion = get_db().execute(
        "SELECT p.id, title, body, created, author_id, username"
        "FROM completion c JOIN user u ON c.user_id = u.id"
        "WHERE p.id = ?",
        (com_id,)
    ).fetchone()

    if completion is None:
        abort(404, f"Completion with id {com_id} doesn't exist.")

    return completion

# @bp.route('/<int:id>/delete', methods=('POST',))
# @login_required
# def delete(com_id: int):
#     """Deletes a completion"""
#     get_completion(com_id)
#     db = get_db()
#     db.execute('DELETE FROM completion WHERE id = ?', (id,))
#     db.commit()
#     return redirect(url_for('completion.index'))
