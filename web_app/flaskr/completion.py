"""Contains the functions for the completion page and the completion blueprint"""

from dataclasses import dataclass
from sqlite3 import Row
from typing import Any, Tuple, Dict, List

from flask import Blueprint, g, render_template, request, redirect, url_for
from werkzeug.datastructures import ImmutableMultiDict

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
    # TODO: fix this method so it does something
    my_db = get_db()
    # Execute a SQL query and return the results
    QUERY = "SELECT * FROM completion c JOIN user u ON c.user_id = u.id ORDER BY created DESC"
    completions: List[Row] = my_db.execute(QUERY).fetchall()
    return render_template("completion/index.html", completions=completions)


@login_required
def add_comp_to_db(comp_data: CompletionData):
    """Adds an answer to the database"""
    QUERY_STRUCTURE: str = """INSERT INTO completion 
                        (user_id, model_id, prompt, answer, num_tokens, generation_type, top_p, top_k, temperature)
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
                 comp_data.num_tokens, str(generator.generation_type),
                 top_p, top_k,
                 generator.temp)
    connection.execute(QUERY_STRUCTURE, arguments)
    connection.commit()


def preprocess_create_form(old_request: ImmutableMultiDict) -> Dict[str, Any]:
    """Preprocesses the data from the create form and returns the processed arguments as a dict"""
    DEFAULTS: Dict[str, Tuple[Any, type]] = {
        "top_p": (None, float),
        "top_k": (None, int),
        "temperature": (1.0, float),
        "num_tokens": (4, int),
        "prompt": ("I had so much fun in the", str),
        "model_name": ("facebook/opt-125m", str),
        "group_size": (2, int),
        "generation_type": ("tree", str)
    }

    new_request: dict = {}

    for input_name, (default_value, wanted_type) in DEFAULTS.items():
        if old_request[input_name] == "":
            new_request[input_name] = default_value
        else:
            new_request[input_name] = wanted_type(old_request[input_name])

    if new_request['top_k'] is None and new_request['top_p'] is None:
        new_request['top_k'] = 1
        new_request['top_p'] = 0.0

    generation_type_names_to_classes: dict = {"sampling": SamplingGenerator, "tree": TreeGenerator}

    if new_request['generation_type'] in generation_type_names_to_classes:
        new_request['generation_type_class'] = generation_type_names_to_classes[new_request['generation_type']]
    else:
        raise ValueError(f"generation_type must be either one of"
                         f" {generation_type_names_to_classes.keys()}, "
                         f"got: {new_request['generation_type']}"
                         f"the full request is {old_request}")
    return new_request


@login_required
@bp.route('/create', methods=('GET', 'POST'))
def create():
    """Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion"""
    if request.method == "POST":
        new_request = preprocess_create_form(request.form)
        text_generator: TextGenerator = new_request['generation_type_class'](
                model_name=new_request['model_name'],
                group_size=new_request['group_size'],
                top_p=new_request['top_p'],
                top_k=new_request['top_k'],
                temp=new_request['temperature']
        )
        answer: str = text_generator(prompt=new_request['prompt'], num_new_tokens=new_request['num_tokens'])
        completion = CompletionData(
            prompt=new_request['prompt'],
            answer=answer,
            num_tokens=new_request['num_tokens'],
            generator=text_generator
        )
        add_comp_to_db(completion)
        return redirect(url_for('completion.index'))
    return render_template("completion/create.html")
