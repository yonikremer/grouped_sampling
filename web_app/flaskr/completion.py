"""Contains the functions for the completion page and the completion blueprint"""
from typing import Union, Optional
from functools import lru_cache, reduce
import tensorflow as tf
from flask import Blueprint, flash, g, redirect, render_template, request, url_for, Response
import os
from sqlite3 import Connection
from langdetect import detect

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import tokenize_and_preprocces, init_tokenizer

bp = Blueprint('completion', __name__)


@bp.route('/')
def index() -> str:
    """The page with all the completions"""
    my_db: Connection = get_db()
    # Execute a SQL query and return the results
    completions: list[dict] = my_db.execute(
        'SELECT c.id, user_id, model_id, created, prompt, answer FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC').fetchall()
    return render_template("completion/index.html", completions = completions)


def combenations(mat: list[list[int]]) -> list[list[int]]:
    """Returns all the lists such that list[j] is in mat[j]
    comlexity: prod([len(mat[i]) for i in range(len(mat))])"""
    if len(mat) == 1:
        return [[mat[0][i]] for i in range(len(mat[0]))]
    else:
        res: list[list[int]] = []
        for i in mat[0]:
            for j in combenations(mat[1:]):
                res.append([i] + j)
        return res


def doesnt_have_duplicates(l: list[int]) -> bool:
    """Return if there isnt a repetition in the list
    complexity: O(n) where n is the length of the list"""
    return len(l) == len(set(l))


def nucleus_sampeling(set_prob: tf.Tensor, top_p: float, top_k: int) -> list[list[int]]:
    """Returns the most probable completions for the prompt as tokens such that
       the sum probabilities of the ith token adds up to less than top_p, there are at maximum top_k
       diffrent tokens each place and the number of tokens in every set is set_size"""
    # set_prob.shape is [1, set_size, vocab_size]
    indeces: list[list[int]] = []
    for i in range(set_prob.shape[1]):
        # token_prob: tf.TensorSpec(shape=[vocab_size], dtype=tf.float)
        token_prob: tf.Tensor = tf.squeeze(set_prob[:, i, :])
        # token_prob.shape is [vocab_size] and token_prob.dtype is float
        sorted_indces: tf.Tensor = tf.argsort(token_prob, direction='DESCENDING')[:top_k]
        # sorted_indces.shape is [vocab_size] and sorted_indces.dtype is int
        total_prob: float = sorted_indces[0].numpy().item()
        curr_indeces: list[int] = [sorted_indces[0].numpy().item()]
        j: int = 1
        while j < top_k:
            total_prob += token_prob[sorted_indces[j]]
            if total_prob <= top_p:
                token_num: int = sorted_indces[j].numpy().item()
                curr_indeces.append(token_num)
                total_prob += token_prob[token_num].numpy().item()
                j += 1
            else:
                break
        indeces.append(curr_indeces)
    new_prompts: list[list[int]] = combenations(indeces)
    return list(filter(doesnt_have_duplicates, new_prompts))


def token_list_to_str(token_list: list[int]) -> str:
    """converts list of tokens to a string"""
    ten_tokens = tf.expand_dims(tf.convert_to_tensor(token_list, dtype=tf.int32), axis=0)
    bytes_ten = tf.squeeze(g.my_tokenizer.detokenize(ten_tokens).to_tensor())
    bytes_list = bytes_ten.numpy().tolist()
    str_list = [b.decode('utf-8') for b in bytes_list]
    return " ".join(str_list).replace(" ##", "").replace("[PAD]", "")


def tokenize_and_preprocces(text: str) -> tf.Tensor:
    """Converts string to tensor"""
    ragged: tf.RaggedTensor = g.my_tokenizer.tokenize(text)[0, :]
    eager: tf.Tensor = ragged.to_tensor(default_value = 0, shape = [None, 1])  # 0 is the value of the padding token
    sqeezed: tf.Tensor = tf.squeeze(eager, axis = 1)
    typed: tf.Tensor = tf.cast(sqeezed, tf.int32)
    start_token = tf.constant([2], dtype=tf.int32)
    edited: tf.Tensor = tf.concat([start_token, typed], axis = 0)
    expended = tf.expand_dims(edited, axis=0)
    return expended


@lru_cache(maxsize = 10)
def create_tar(set_size: int):
    """Creates a tensor of shape [1, set_size] with random integers between 5 and the size of the vocabulary"""
    return tf.random.uniform([1, set_size], minval=5, maxval=len(g.vocab), dtype=tf.int32)


def complete(model: tf.keras.Model, prompt: Optional[str], top_p: float, top_k: int, num_sets: int, set_size: int) -> list[str]:
    """preprocess the prompt and completes the text"""
    if prompt is None:
        return []
    tokenized_prompt: tf.Tensor = tokenize_and_preprocces(prompt)
    # vocab[0:4] are the saved tokens
    tar = create_tar(set_size)
    # dont incluse the probability for the [PAD] token
    set_prob: tf.Tensor = model([tokenized_prompt, tar], training=False)
    tokenized_ans_list: list[list[int]] = nucleus_sampeling(set_prob, top_p, top_k)
    # print(tokenized_ans_list)
    ans_list: list[str] = [token_list_to_str(tokenized_ans_list[i]) for i in range(len(tokenized_ans_list))]
    if num_sets == 1:
        return ans_list
    new_prompts: list[str] = [f"{prompt} {ans}" for ans in ans_list]
    for index, a_new_prompt in enumerate(new_prompts):
        if "[END]" in a_new_prompt:
            end_index: int = a_new_prompt.index("[END]")
            a_new_prompt = new_prompts[index] = a_new_prompt[:end_index]
    new_answers: list[list[str]] = [complete(new_prompt, top_p, top_k, num_sets - 1) for new_prompt in new_prompts]
    # print(new_answers)
    if len(new_answers) > 1:
        flaten_ans = reduce(lambda x,y: x+y, new_answers)
        # print(flaten_ans)
        return flaten_ans
    if len(new_answers) == 1 and len(new_answers[0]) > 0: return new_answers[0]
    return []



def add_answer(database: Connection, model_id: int, prompt: str, answer: str) -> None:
    """Adds an answer to the database"""
    database.execute('INSERT INTO completion (model_id, prompt, answer, user_id) VALUES (?, ?, ?, ?)',
                    (model_id, prompt, answer, g.user['id']))
    database.commit()


def create_page_errors(model_id_str, prompt, top_k_str, top_p_str, num_sets) -> str:
    """Checks the inputs for the create page and return thr errors"""
    errors: str = ""
    top_k = int(top_k_str)
    top_p_float = float(top_p_str)
    num_sets_int = int(num_sets)
    model_id = int(model_id_str)
    if top_k < 1: errors += "top_k must be greater than 0\n"
    if str(top_k) != top_k_str: errors += "top_k must be an integer\n"
    if not top_p_str: errors += 'Top-P is required\n'
    if str(float(top_p_str)) != top_p_str: errors += 'Top-P must be a float\n'
    if not (0.0 < top_p_float < 1.0): errors += 'Top-P must be between 0 and 1\n' 
    if not num_sets: errors += 'Number of sets is required\n'
    if str(num_sets_int) != num_sets: errors += 'Number of sets must be an integer\n'
    if num_sets_int < 1: errors += 'Number of sets must be greater than 0\n'
    if not prompt: errors += 'Prompt is required.\n'
    if not isinstance(prompt, str): errors += 'prompt must be a string.\n'
    if detect(prompt) != 'en': errors += 'prompt must be in english.\n'
    if not model_id_str: errors += 'You must select a model.\n'
    if int(model_id_str) != model_id or model_id < 0: errors += 'Model id must be aa integer greater than 0.\n'
    return errors

@login_required
@bp.route('/create', methods = ('GET', 'POST'))
def create() -> Union[str, Response]:
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == "POST":
        top_p_str: str = request.form['top_p']
        top_k_str: str = request.form['top_k']
        num_sets: str = request.form['num_sets']
        prompt: str = request.form['prompt']
        model_id_str: str = request.form['model_id']
        errors: str = create_page_errors(model_id_str, prompt, top_k_str, top_p_str, num_sets)
        num_sets_int = int(num_sets)
        model_id = int(model_id_str)
        top_p_float = float(top_p_str)
        top_k = int(top_k_str)
        if not hasattr(g, 'my_tokenizer'):
            my_tokenizer, vocab = init_tokenizer()
            g.my_tokenizer = my_tokenizer
            g.vocab = vocab
        if errors != "":
            flash(errors)
        else:
            my_db: Connection = get_db()
            model_name, set_size = my_db.execute('SELECT model_name, set_size FROM model WHERE id = ?', (model_id,)).fetchone()
            if model_name is None or not isinstance(model_name, str):
                errors += f'Model: {model_id} does not exist./n'
            path: str = f"/flaskr/static/models/{model_name}.pb"
            if not os.path.exists(path):
                errors += f"Model {model_name} does not exist or have a wrong name"
            model: tf.keras.model.Model = tf.saved_model.load(path)
            if errors == "":
                answers: list[str] = complete(prompt, model, top_p_float, top_k, num_sets_int, set_size)
                if len(answers) == 0:
                    errors += "No valid completions suggested, please try a diffrent model"
                if not isinstance(answers[0], str) or not isinstance(answers, list):
                    errors += 'Something with the model is not working as intended. Please try a different model./n'
                if errors == "":
                    for ans in answers: add_answer(my_db, model_id, prompt, ans)
                    return redirect(url_for('completion/index.html'), )
    return render_template("completion/create.html")
