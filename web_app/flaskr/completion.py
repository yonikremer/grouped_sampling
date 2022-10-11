"""Contains the functions for the completion page and the completion blueprint"""

from dataclasses import dataclass

from flask import Blueprint, g, render_template, request, redirect, url_for

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
                        (user_id, model_id, prompt, answer, num_tokens, generation_type, top_p, top_k, temprature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    connection = get_db()
    generator: TextGenerator = comp_data.generator
    model_id: int = get_model_id(generator.model_name)

    if hasattr(generator, "top_k"):
        top_k = generator.top_k
    else:
        top_k = None
    if hasattr(generator, "top_p"):
        top_p = generator.top_p
    else:
        top_p = None
    arguments = (g.user['id'], model_id, comp_data.prompt, comp_data.answer,
                 comp_data.num_tokens, generator.generation_type,
                 top_p, top_k,
                 generator.temp)
    connection.execute(STRUCTURE, arguments)
    connection.commit()


@login_required
@bp.route('/create', methods=('GET', 'POST'))
def create():
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == "POST":
        if request.form['top_p'] == "":
            top_p = None
        else:
            top_p = float(request.form['top_p'])
        if request.form['top_k'] == "":
            top_k = None
        else:
            top_k = int(request.form['top_k'])

        num_tokens = int(request.form['num_tokens'])

        prompt = request.form['prompt']

        model_name = request.form['my_model_name']

        group_size = int(request.form['group_size'])

        generation_type_name = request.form['generation_type']

        if request.form['temperature'] == "":
            temperature = 1.0
        else:
            temperature = float(request.form['temperature'])

        generation_type_names_to_classes = {"sampling": SamplingGenerator, "tree": TreeGenerator}

        if generation_type_name in generation_type_names_to_classes:
            text_generator_type = generation_type_names_to_classes[generation_type_name]
        else:
            raise ValueError(f"generation_type must be either one of"
                             f" {generation_type_names_to_classes.keys()}, "
                             f"got: {generation_type_name}"
                             f"the full request is {request.form}")
        text_generator: TextGenerator = text_generator_type(
                model_name=model_name,
                group_size=group_size,
                top_p=top_p,
                top_k=top_k,
                temp=temperature
        )
        print("successfully created a text generator")
        answer: str = text_generator(prompt, num_tokens)
        print("successfully generated an answer")
        completion = CompletionData(prompt, answer, num_tokens, text_generator)
        add_comp_to_db(completion)
        print("successfully added the completion to the database")
        return redirect(url_for('completion/index.html'))
    return render_template("completion/create.html")
