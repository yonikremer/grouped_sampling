"""Contains the functions for the completion page and the completion blueprint"""
import os
from typing import List, Tuple, Dict

import tensorflow as tf
from flask import Blueprint, g, redirect, render_template, request, url_for, flash
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import Softmax

from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import tf_preprocess_and_tokenize, init_tf_tokenizer, create_tar
from flaskr.sampling import grouped_sampling, seq_prob


bp = Blueprint('completion', __name__)


@bp.route('/')
def index():
    """The page with all the completions"""
    my_db = get_db()
    # Execute a SQL query and return the results
    query = "SELECT * FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC"
    completions = my_db.execute(query).fetchall()
    return render_template("completion/index.html", completions = completions)


def get_prob_mat(model_name, prompt, group_size):
    """Returns the probability matrix as a list of lists of floats"""
    model = g.loaded_models[model_name]
    tokenizer = g.loaded_tokenizers[model_name]
    if isinstance(model, AutoModelForCausalLM): 
        inputs = tokenizer(prompt, return_tensors="pt")
        if "bloom" in model_name:
            logits_pt_tensor = model(**inputs, labels=inputs["input_ids"]).logits.squeeze(0)
        else:
            logits_pt_tensor = model(**inputs).logits.squeeze(0)
        prob_tensor = Softmax(dim=1)(logits_pt_tensor)
        try:
            prob_tensor = prob_tensor[:group_size, :]
        except IndexError:
            pass
        
        prob_mat = [prob_tensor[i, :].tolist() for i in range(group_size)]
    else:
        tokenized_prompt = tf_preprocess_and_tokenize(prompt)
        tar = create_tar(group_size)
        prob_ten = model([tokenized_prompt, tar], training=False)
        prob_squeezed = tf.squeeze(prob_ten)
        prob_ndarray = tf.make_ndarray(prob_squeezed)
        prob_mat = [prob_ndarray[i, :].tolist() for i in range(group_size)]
    return prob_mat


def flatten(l: list) -> List:
    """Gets a list where some of the elements might be lists
    and adds every item in the inner list to the outer list.
    example: [1, [2, 3], 4, [[5]]] -> [1, 2, 3, 4, 5]
    Complexity: O(len(the flatten list) + the number of diffrent lists))"""
    new_list = []
    for item in l:
        if isinstance(item, list):
            new_list.extend(flatten(item))
        else:
            new_list.append(item)
    return new_list


def complete(model_name, org_prompt, top_p, top_k, num_groups, group_size, org_prompt_prob = 1.0) -> Dict[Tuple[int], float]:
    """preprocess the prompt and completes the text
    model name: str
    org_prompt: str
    top_p: float (0.0 - 1.0)
    top_k: int > 0
    num_groups: int > 0
    group_size: int > 0
    org_prompt_prob: float <= 1
    """
    tokenizer = g.loaded_tokenizers[model_name]
    if isinstance(org_prompt, str):
        str_prompt = org_prompt
        if "bloom" in model_name:
            tokenized_prompt_ten = tokenizer(str_prompt, return_tensors="pt")["input_ids"]
        else:
            tokenized_prompt_ten = tokenizer(str_prompt, return_tensors="pt")
        tokenized_prompt_list = tokenized_prompt_ten.tolist()
    elif isinstance(org_prompt, list):
        tokenized_prompt_list = flatten(org_prompt)
    else:
        raise ValueError("org_prompt must be a string or list of integers")
    
    prob_mat = get_prob_mat(model_name, str_prompt, group_size)
    tokenized_ans_list = grouped_sampling(prob_mat, top_p, top_k)
    prob_list: List[Tuple[List[int], float]] = [seq_prob(seq, prob_mat, org_prompt_prob) for seq in tokenized_ans_list]
    new_prompts: List[List[int]] = [tokenized_prompt_list + ans for ans in tokenized_ans_list]
    complition_prob_dict = remove_duplicates(new_prompts, prob_list)
    if num_groups == 1:
        return complition_prob_dict
    new_complitions: Dict[Tuple[int], float] = dict()
    for curr_new_prompt, curr_new_prompt_prob in complition_prob_dict.items():
        curr_completions: Dict[Tuple[int], float] = complete(model_name, curr_new_prompt, top_p, top_k, num_groups - 1, group_size, curr_new_prompt_prob)
        tokens: Tuple[int]
        prob: float
        for tokens, prob in curr_completions.items():
            new_complitions[tokens] = prob
    return new_complitions


def remove_duplicates(completions: List[List[int]], probs: List[float]) -> Dict[Tuple[int], float]:
    """Given a list of tokenized answers and the probability of each complition,
    removes every repeated completion and every complition that have repeated tokens"""
    filtered_completions: Dict[Tuple[int], float] = dict()
    for curr_comp, curr_prob in zip(completions, probs):
        if len(curr_comp) == len(set(curr_comp)):
            curr_comp_tuple = tuple(curr_comp)
            filtered_completions[curr_comp_tuple] = curr_prob
    return filtered_completions


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
