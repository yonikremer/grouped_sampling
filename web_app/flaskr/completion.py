"""Contains the functions for the completion page and the completion blueprint"""

from dataclasses import dataclass

from flask import Blueprint, g, render_template, request

from text_generator import TextGenerator
from sampling_generator import SamplingGenerator
from tree_generator import TreeGenerator
from .auth import login_required
from .database import get_db
from .model import get_model_id

bp = Blueprint('completion', __name__)


@dataclass
class CompletionData:
    """Contains all the data for the completion page"""
    prompt: str
    answer: str
    num_tokens: int
    generator: TextGenerator


@bp.route('/')
def index():
    """The page with all the completions"""
    my_db = get_db()
    # Execute a SQL query and return the results
    query = "SELECT * FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC"
    completions = my_db.execute(query).fetchall()
    return render_template("completion/index.html", completions=completions)


def add_comp_to_db(comp_data: CompletionData):
    """Adds an answer to the database"""
    STRUCTURE: str = """INSERT INTO completion 
                        (user_id, model_id, prompt, answer, num_tokens, generation_type, TOP_P, TOP_K, temprature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    connection = get_db()
    generator: TextGenerator = comp_data.generator
    model_id: int = get_model_id(generator.model_name)
    arguments = (g.user['id'], model_id, comp_data.prompt, comp_data.answer,
                 comp_data.num_tokens, generator.generation_type,
                 generator.top_p, generator.top_k,
                 generator.temperature)
    connection.execute(STRUCTURE, arguments)
    connection.commit()


@login_required
@bp.route('/create', methods=('GET', 'POST'))
def create():
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == "POST":
        print("called function create")
        top_p = request.form['TOP_P']
        top_k = request.form['TOP_K']
        num_tokens = request.form['num_tokens']
        prompt = request.form['prompt']
        model_name = request.form['model_name']
        group_size = request.form['group_size']
        generation_type = request.form['generation_type']
        temperature = request.form['temperature']
        raise NotImplementedError
        # TODO: create a generator based on the form data
        # answer: str = generator(prompt, num_tokens)
        # completion = CompletionData(prompt, answer, num_tokens, generator)
        # add_comp_to_db(completion)
        # return redirect(url_for('completion/index.html'))
    return render_template("completion/create.html")
