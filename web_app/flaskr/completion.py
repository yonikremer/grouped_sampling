"""Contains the functions for the completion page and the completion blueprint"""
from functools import lru_cache
import os
from typing import List, Iterable, Tuple

import tensorflow as tf
from langdetect import detect
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from transformers import AutoTokenizer, AutoModel
from torch.nn import Softmax

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import tokenize_and_preprocess, init_tokenizer, token_list_to_str, create_tar
from flaskr.sampling import grouped_sampling


bp = Blueprint('completion', __name__)


@bp.route('/')
def index():
    """The page with all the completions"""
    my_db = get_db()
    # Execute a SQL query and return the results
    query = "SELECT * FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC"
    completions= my_db.execute(query).fetchall()
    return render_template("completion/index.html", completions = completions)


def get_prob_mat(model_name, prompt, group_size):
    """Returns the probability matrix as a list of lists of floats"""
    is_auto_model = True
    try:
        model = AutoModel.from_pretrained(model_name)
    except:
        is_auto_model = False
    if is_auto_model: 
        if not hasattr(g, "pt_tokenizer"):
            g.pt_tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = g.pt_tokenizer.tokenize(prompt, return_tensors="pt")
        outputs = model(**inputs, labels=inputs["input_ids"])
        pt_softmax_func = Softmax(dim=1)
        prob_tensor = pt_softmax_func(outputs.logits)
        prob_tensor.squeeze(0)
        prob_tensor = prob_tensor[:group_size, :]
        prob_mat = [prob_tensor[i, :].tolist() for i in range(group_size)]
    else:
        if not hasattr(g, 'tf_tokenizer'):
            tf_tokenizer, vocab = init_tokenizer()
            g.tf_tokenizer = tf_tokenizer
            g.vocab_size = len(vocab)
        model_path = f"/flaskr/static/models/{model_name}.pb"
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model {model_name} not found")
        model = tf.saved_model.load(model_path)
        tokenized_prompt = tokenize_and_preprocess(prompt)
        tar = create_tar(group_size)
        prob_ten = model([tokenized_prompt, tar], training=False)
        prob_squeezed = tf.squeeze(prob_ten)
        prob_ndarray = tf.make_ndarray(prob_squeezed)
        prob_mat = [prob_ndarray[i, :].tolist() for i in range(group_size)]
    return prob_mat


def complete(model_name, org_prompt, top_p, top_k, num_groups, group_size, org_prompt_prob = 1.0) -> Tuple[List[str], List[float]]:
    """preprocess the prompt and completes the text"""
    prob_mat = get_prob_mat(model_name, org_prompt, group_size)
    tokenized_ans_list, prob_list = grouped_sampling(prob_mat, top_p, top_k, group_size)

    if "bloom" in model_name.lower():
        library = "pytorch"
    else:
        library = "tensorflow"

    ans_list: List[str] = [token_list_to_str(ans, library) for ans in tokenized_ans_list]
    new_prompts: List[str] = [f"{org_prompt} {ans}" for ans in ans_list]
    if num_groups == 1:
        return new_prompts, prob_list
    for i, curr_new_prompt in enumerate(new_prompts):
        if "[END]" in curr_new_prompt:
            end_index = curr_new_prompt.index("[END]")
            new_prompts[i] = curr_new_prompt[:end_index]
    all_new_completions: List[str] = []
    all_new_probs: List[float] = []
    for curr_new_prompt, curr_new_prompt_prob in zip(new_prompts, prob_list):
        curr_new_prompt_prob *= org_prompt_prob
        curr_completions, curr_probs = complete(model_name, curr_new_prompt, top_p, top_k, num_groups - 1, group_size, curr_new_prompt_prob)
        all_new_completions.extend(curr_completions)
        all_new_probs.extend(curr_probs)
    return all_new_completions, all_new_probs


def add_answer_to_db(connection, model_id, prompt, answer):
    """Adds an answer to the database"""
    query_structure = "INSERT INTO completion (model_id, prompt, answer, user_id) VALUES (?, ?, ?, ?)"
    connection.execute(query_structure, (model_id, prompt, answer, g.user['id']))
    connection.commit()


def create_page_errors(model_id_str, prompt, top_k_str, top_p_str, num_groups):
    """Checks the inputs for the create page and return thr errors"""
    errors = ""
    top_k = int(top_k_str)
    top_p_float = float(top_p_str)
    num_groups_int = int(num_groups)
    model_id = int(model_id_str)
    if top_k < 1:
        errors += "top_k must be greater than 0\n"
    if str(top_k) != top_k_str:
        errors += "top_k must be an integer\n"
    if not top_p_str:
        errors += 'Top-P is required\n'
    if str(float(top_p_str)) != top_p_str:
        errors += 'Top-P must be a float\n'
    if not (0.0 <= top_p_float <= 1.0):
        errors += 'Top-P must be between 0 and 1\n'
    if not num_groups:
        errors += 'Number of groups is required\n'
    if str(num_groups_int) != num_groups:
        errors += 'Number of groups must be an integer\n'
    if num_groups_int < 1:
        errors += 'Number of groups must be greater than 0\n'
    if not prompt:
        errors += 'Prompt is required.\n'
    if not isinstance(prompt, str):
        errors += 'prompt must be a string.\n'
    if detect(prompt) != 'en':
        errors += 'prompt must be in english.\n'
    if not model_id_str:
        errors += 'You must select a model.\n'
    if int(model_id_str) != model_id or model_id < 0:
        errors += 'Model id must be aa integer greater than 0.\n'
    return errors


@login_required
@bp.route('/create', methods = ('GET', 'POST'))
def create():
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == "POST":
        top_p_str = request.form['top_p']
        top_k_str = request.form['top_k']
        num_groups = request.form['num_groups']
        prompt = request.form['prompt']
        model_id_str = request.form['model_id']
        errors = create_page_errors(model_id_str, prompt, top_k_str, top_p_str, num_groups)
        num_groups_int = int(num_groups)
        model_id = int(model_id_str)
        top_p_float = float(top_p_str)
        top_k = int(top_k_str)
        if errors != "":
            flash(errors)
        else:
            my_db = get_db()
            model_name, group_size = my_db.execute('SELECT model_name, group_size FROM model WHERE id = ?', (model_id,)).fetchone()
            if model_name is None or not isinstance(model_name, str):
                errors += f'Model: {model_id} does not exist./n'
            completions, probabilities = complete(model_name, prompt, top_p_float, top_k, num_groups_int, group_size)
            if sum(probabilities) > 1:
                errors += "The probabilities of the completions are not normalized.\n"
            if errors == "":
                max_prob = max(probabilities)
                index_of_best_completion = probabilities.index(max_prob)
                best_completion = completions[index_of_best_completion]
                add_answer_to_db(my_db, model_id, prompt, best_completion)
                return redirect(url_for('completion/index.html'))
    return render_template("completion/create.html")
