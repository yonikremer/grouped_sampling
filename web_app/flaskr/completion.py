"""Contains the functions for the completion page and the completion blueprint"""

from flask import Blueprint, g, redirect, render_template, request, url_for, flash

from sample_generator import SampleGenerator
from tree_generator import TreeGenerator
from flaskr.auth import login_required
from flaskr.db import get_db
from flaskr.tokenizer import init_tf_tokenizer


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
        top_p = request.form['top_p']
        top_k = request.form['top_k']
        num_tokens = request.form['num_tokens']
        prompt = request.form['prompt']
        model_name = request.form['model_name']
        group_size = request.form['group_size']
        generation_type = request.form['generation_type']
        temperature = request.form['temperature']
        
        add_answer_to_db(model_name, prompt, answer)
        if generation_type == "tree":
            generator = TreeGenerator(model_name, group_size, top_k, top_p, temperature)
        elif generation_type == "sampling":
            generator = SampleGenerator(model_name, group_size, top_k, top_p, temperature)
        answer: str = generator(prompt, num_tokens) 
        add_answer_to_db(model_name, prompt, answer)
        return redirect(url_for('completion/index.html'))
    return render_template("completion/create.html")
