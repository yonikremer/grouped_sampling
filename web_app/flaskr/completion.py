"""Contains the functions for the completion page and the completion blueprint"""
from functools import lru_cache, reduce
import os

import tensorflow as tf
from keras import Model as KerasModel
from langdetect import detect
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from transformers import BloomTokenizerFast, BloomForCausalLM
from torch.nn import Softmax

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import tokenize_and_preprocess, init_tokenizer


bp = Blueprint('completion', __name__)


@bp.route('/')
def index():
    """The page with all the completions"""
    my_db = get_db()
    # Execute a SQL query and return the results
    query = "SELECT * FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC"
    completions= my_db.execute(query).fetchall()
    return render_template("completion/index.html", completions = completions)


def combinations(mat):
    """Returns all the lists such that list[j] is in mat[j]
    complexity: prod([len(mat[i]) for i in range(len(mat))])"""
    if len(mat) == 1:
        return [[mat[0][i]] for i in range(len(mat[0]))]
    res = []
    for i in mat[0]:
        for j in combinations(mat[1:]):
            res.append([i] + j)
    return res


def doesnt_have_duplicates(my_list):
    """Return if there isn't a repetition in the list
    complexity: O(n) where n is the length of the list"""
    return len(my_list) == len(set(my_list))


def grouped_sampling(prob_mat, top_p, top_k, group_size):
    """The same as grouped_sampling_tf but works with pytorch"""
    
    # prob_tensor.shape is now (group_size, vocab_size)
    indices = []
    for i in range(group_size):
        token_prob = prob_mat[i]
        vocab_size = len(token_prob)
        indexed_prob = list(zip(token_prob, range(vocab_size)))
        sorted_indexed_prob = sorted(indexed_prob, key=lambda x: x[0], reverse=True)[:top_k]
        total_prob = 0
        curr_indices = []
        for prob, token in sorted_indexed_prob:
            if total_prob + prob > top_p:
                break
            total_prob += prob
            curr_indices.append(token)
        indices.append(curr_indices)
    new_prompts = combinations(indices)
    return list(filter(doesnt_have_duplicates, new_prompts))


def token_list_to_str(token_list):
    """converts list of tokens to a string"""
    ten_tokens = tf.expand_dims(tf.convert_to_tensor(token_list, dtype=tf.int32), axis=0)
    bytes_ten = tf.squeeze(g.my_tokenizer.detokenize(ten_tokens).to_tensor())
    bytes_list = bytes_ten.numpy().tolist()
    str_list = [b.decode('utf-8') for b in bytes_list]
    return " ".join(str_list).replace(" ##", "").replace("[PAD]", "")


@lru_cache(maxsize = 10)
def create_tar(group_size):
    """Creates a tensor of shape [1, group_size]
    with random integers between 5 and the size of the vocabulary"""
    return tf.random.uniform([1, group_size], minval=5, maxval=len(g.vocab), dtype=tf.int32)


def complete(model_name, prompt, top_p, top_k, num_groups, group_size):
    """preprocess the prompt and completes the text"""
    if "bloom" in model_name.lower():
        model_path = f"bigscience/{model_name}"
        model_name = BloomForCausalLM.from_pretrained(model_path)
        pt_tokenizer = BloomTokenizerFast.from_pretrained(model_path)
        inputs = pt_tokenizer(prompt, return_tensors="pt")
        outputs = model_name(**inputs, labels=inputs["input_ids"])
        prob_tensor = Softmax(outputs.logits)
        prob_tensor.squeeze(0)
        prob_tensor = prob_tensor[:group_size, :]
        prob_mat = [prob_tensor[i, :].tolist() for i in range(group_size)]
    else:
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

    prob_mat = [prob_mat[i, :].tolist() for i in range(group_size)]
    tokenized_ans_list = grouped_sampling(prob_mat, top_p, top_k, group_size)
    ans_list = [token_list_to_str(tokenized_ans_list[i]) for i in range(len(tokenized_ans_list))]
    if num_groups == 1:
        return ans_list
    new_prompts = [f"{prompt} {ans}" for ans in ans_list]
    for i, new_prompt in enumerate(new_prompts):
        if "[END]" in new_prompt:
            end_index = new_prompt.index("[END]")
            new_prompts[i] = new_prompt[:end_index]
    new_answers = [complete(model_name, prompt, top_p, top_k, num_groups - 1, group_size) for prompt in new_prompts]
    if len(new_answers) > 1:
        flatten_ans = reduce(lambda x, y: x + y, new_answers)
        return flatten_ans
    if len(new_answers) == 1 and len(new_answers[0]) > 0:
        return new_answers[0]
    return []


def add_answer(database, model_id, prompt, answer):
    """Adds an answer to the database"""
    query_structure = "INSERT INTO completion (model_id, prompt, answer, user_id) VALUES (?, ?, ?, ?)"
    database.execute(query_structure, (model_id, prompt, answer, g.user['id']))
    database.commit()


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
        if not hasattr(g, 'tf_tokenizer'):
            tf_tokenizer, vocab = init_tokenizer()
            g.tf_tokenizer = tf_tokenizer
            g.vocab = vocab
        if errors != "":
            flash(errors)
        else:
            my_db = get_db()
            model_name, group_size = my_db.execute('SELECT model_name, group_size FROM model WHERE id = ?', (model_id,)).fetchone()
            if model_name is None or not isinstance(model_name, str):
                errors += f'Model: {model_id} does not exist./n'
            completions = complete(model_name, prompt, top_p_float, top_k, num_groups_int, group_size)
            if errors == "":
                if len(completions) == 0:
                    errors += "No valid completions suggested, please try bigger pot p and k"
                if not isinstance(completions[0], str) or not isinstance(completions, list):
                    errors += 'Something with the model is not working. Please try a different model./n'
                if errors == "":
                    for ans in completions:
                        add_answer(my_db, model_id, prompt, ans)
                    return redirect(url_for('completion/index.html'), )
    return render_template("completion/create.html")
