"""Contains the functions for the completion page and the completion blueprint"""
from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any, Dict, Tuple

import pandas as pd
from flask import Blueprint, g, redirect, render_template, request, url_for
from werkzeug.datastructures import ImmutableMultiDict

from src.grouped_sampling import (
    CompletionDict,
    GroupedGenerationPipeLine,
    GroupedSamplingPipeLine,
    GroupedTreePipeLine,
    is_supported,
    unsupported_model_name_error_message,
)

from .auth import login_required
from .database import get_db
from .model import get_model_id

bp = Blueprint("completion", __name__)


@dataclass
class CompletionData:
    """Contains all the data for the completion page"""

    prompt: str
    answer: str
    num_tokens: int
    generator: GroupedGenerationPipeLine


@bp.route("/")
def index():
    """The page with all the completions"""
    my_db: Connection = get_db()
    # Execute a SQL query and return the results
    query = "SELECT " \
            "username, c.created, prompt, answer, num_tokens, model_name," \
            " group_size, generation_type, top_p, top_k, temperature " \
            "FROM " \
            "completion c JOIN user u ON c.user_id = u.id JOIN model m ON c.model_id = m.id  ORDER BY c.created"
    df = pd.read_sql_query(query, my_db)
    html_table = df.to_html(classes="data",
                            header="true",
                            border=0,
                            index=False)
    return render_template("completion/index.html", table=html_table)


@login_required
def add_comp_to_db(comp_data: CompletionData):
    """Adds an answer to the database"""
    query_structure: str = (
        "INSERT INTO completion ("
        "user_id, prompt, answer, num_tokens, model_id, group_size, generation_type, top_p, top_k, temperature"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )
    connection = get_db()
    generator: GroupedGenerationPipeLine = comp_data.generator
    model_id: int = get_model_id(generator.model_name)

    if hasattr(generator, "top_k"):
        top_k = generator.top_k
    else:
        top_k = None
    if hasattr(generator, "top_p"):
        top_p = generator.top_p
    else:
        top_p = None
    #     user_id, prompt, answer, num_tokens, model_id, group_size, generation_type, top_p, top_k, temperature
    arguments = (
        g.user["id"],
        comp_data.prompt,
        comp_data.answer,
        comp_data.num_tokens,
        model_id,
        generator.wrapped_model.group_size,
        generator.generation_type.value,
        top_p,
        top_k,
        generator.wrapped_model.temp,
    )
    connection.execute(query_structure, arguments)
    connection.commit()


def preprocess_create_form(old_request: ImmutableMultiDict) -> Dict[str, Any]:
    """Preprocesses the data from the create form and returns the processed arguments as a dict"""
    defaults: Dict[str, Tuple[Any, type]] = {
        "top_p": (None, float),
        "top_k": (None, int),
        "temperature": (1.0, float),
        "num_tokens": (4, int),
        "prompt": ("I had so much fun in the", str),
        "model_name": ("facebook/opt-125m", str),
        "group_size": (2, int),
        "generation_type": ("tree", str),
    }

    new_request: dict = {}

    for input_name, (default_value, wanted_type) in defaults.items():
        if old_request[input_name] == "":
            new_request[input_name] = default_value
        else:
            new_request[input_name] = wanted_type(old_request[input_name])

    generation_type_names_to_classes = {
        "sampling": GroupedSamplingPipeLine,
        "tree": GroupedTreePipeLine,
    }

    if new_request["generation_type"] in generation_type_names_to_classes:
        new_request[
            "generation_type_class"] = generation_type_names_to_classes[
            new_request["generation_type"]
        ]
    else:
        raise ValueError(f"generation_type must be either one of"
                         f" {generation_type_names_to_classes.keys()}, "
                         f"got: {new_request['generation_type']}"
                         f"the full request is {old_request}")
    return new_request


@login_required
@bp.route("/create", methods=("GET", "POST"))
def create():
    """
    Creates a new completion
    If a completion is created successfully, directs to the main page.
    else, recursively redirect to this page (completion/create) until a successful completion
    """
    if request.method == "POST":
        raw_request = preprocess_create_form(request.form)
        print("Checking if model is supported")
        if not is_supported(raw_request["model_name"]):
            error = unsupported_model_name_error_message(
                raw_request["model_name"])
            return render_template("completion/create.html", error=error)
        pipeline: GroupedGenerationPipeLine = raw_request[
            "generation_type_class"
        ](
            load_in_8bit=False,
            model_name=raw_request["model_name"],
            group_size=raw_request["group_size"],
            top_p=raw_request["top_p"],
            top_k=raw_request["top_k"],
            temp=raw_request["temperature"],
        )
        raw_answer: CompletionDict = pipeline(
            prompt_s=raw_request["prompt"],
            max_new_tokens=raw_request["num_tokens"],
            return_text=True,
            return_tensors=False,
            return_full_text=False,
        )
        answer: str = raw_answer["generated_text"]
        completion = CompletionData(
            prompt=raw_request["prompt"],
            answer=answer,
            num_tokens=raw_request["num_tokens"],
            generator=pipeline,
        )
        add_comp_to_db(completion)
        return redirect(url_for("completion.index"))
    return render_template("completion/create.html")
