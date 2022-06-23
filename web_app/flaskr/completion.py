"""Contains the functions for the completion page and the completion blueprint"""
from typing import Optional, List, Union, Set
import tensorflow as tf
from flask import Blueprint, flash, g, redirect, render_template, request, url_for, Response

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import tokenize, init_tokenizer
from werkzeug.exceptions import abort
import os
from sqlite3 import Connection

bp = Blueprint('completion', __name__)


@bp.route('/')
def index() -> str:
    """The page with all the completions"""
    my_db: Connection = get_db()
    # Execute a SQL query and return the results
    completions: List[dict] = my_db.execute(
        'SELECT c.id, user_id, model_id, created, prompt, answer FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC').fetchall()
    return render_template("completion/index.html", completions = completions)


def complete(prompt: str, model_name: str) -> str:
    """Loads the model, preprocess the prompt and completes the text"""
    if not hasattr(g, 'my_tokenizer'):
        my_tokenizer = init_tokenizer()
        g.my_tokenizer = my_tokenizer
    path: str = f"/flaskr/static/models/{model_name}.pb"
    if not os.path.exists(path):
        flash(f"Model {model_name} does not exist or have a wrong name")
        return render_template("completion/create.html")
    model: tf.keras.model.Model = tf.saved_model.load(path)
    tokenized_prompt: tf.Tensor = tokenize(prompt)
    if tokenized_prompt.shape[0] > g.MAX_LEN:
        flash('The prompt is too long')
        return render_template("completion/create.html")
    tokenized_ans: tf.TensorSpec(shape=[1, None, None]) = model.predict(tokenized_prompt)
    if not tokenized_ans.shape[0] == 1:
        flash('The model have a wrong output shape')
        return render_template("completion/create.html")
    if not tokenized_ans.dtype in (tf.float32, tf.float16):
        flash('The model have a wrong output data type')
        return render_template("completion/create.html")
    string_tensor_ans: tf.RaggedTensor = g.tokenizer.detokenize(tokenized_ans)
    string_words: List[str] = []
    for one_d_tensor in string_tensor_ans:
        zero_d_tensor: tf.Tensor
        for zero_d_tensor in one_d_tensor:
            encoded = str(zero_d_tensor.numpy())  # "b'the string'"
            string_word = encoded[2:-1]  # "the string"
            string_words.append(string_word)

    sentence: str = ' '.join(string_words)
    if not isinstance(sentence, str):
        flash('Something with the model is not working as intended. Please try a different model')
    remove_words: Set[str] = {'[PAD]', '[UNK]', '[START]', '[END]', '[MASK]'}
    for word in remove_words:
        sentence = sentence.replace(word, '')
    return sentence


@login_required
@bp.route('/create', methods = ('GET', 'POST'))
def create() -> Union[str, Response]:
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == 'POST':
        prompt: str = request.form['prompt']
        model_id_str: str = request.form['model_id']
        model_id: int = int(model_id_str)
        error: Optional[str] = None
        if not prompt:
            error = 'Prompt is required.'
        if not isinstance(prompt, str):
            error = 'prompt must be a string.'
        if not model_id_str:
            error = 'You must select a model.'
        if int(str(model_id)) != model_id or model_id < 0:
            error = 'Model id must be aa integer greater than 0'
        if error is not None:
            flash(error)
        else:
            my_db: Connection = get_db()
            model_name: str = my_db.execute('SELECT model_name FROM model WHERE id = ?', (model_id,)).fetchone()
            if model_name is None or not isinstance(model_name, str):
                error = f'Model: {model_id} does not exist'
            if error is None:
                answer: str = complete(prompt, model_name)
                if not isinstance(answer, str):
                    error = 'Something with the model is not working as intended. Please try a different model'
                if error is None:
                    my_db.execute(
                        'INSERT INTO completion (model_id, prompt, answer, user_id)'
                        ' VALUES (?, ?, ?, ?)',
                        (model_id, prompt, answer, g.user['id'])
                    )
                    my_db.commit()
                    return redirect(url_for('completion/index.html'), )
    return render_template("completion/create.html")
