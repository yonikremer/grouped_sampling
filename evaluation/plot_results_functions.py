"""The purpose of this file is to refactor code from the plot_results.ipynb notebook into a python file"""
from typing import List, Dict, Optional

from IPython.core.display import Markdown, display
from comet_ml import API, APIExperiment
from matplotlib import pyplot as plt

from helpers import STAT_NAME_TO_FUNC, BERT_SCORES, get_project_name, get_comet_api_key

api = API(api_key=get_comet_api_key())
stat_names = tuple(stat_name for stat_name, _ in STAT_NAME_TO_FUNC)
metric_names = BERT_SCORES
USERNAME: str = "yonikremer"
X_LABEL = "group size"
FIGURE_SIZE = (10, 7.5)

plt.autoscale(False)


def experiment_filter(exp: APIExperiment) -> bool:
    try:
        return get_parameter(exp, "answer_length_multiplier") == "1.25"
    except RuntimeError:
        return False


def get_relevant_experiments() -> List[APIExperiment]:
    all_experiments: List[APIExperiment] = api.get_experiments(
        workspace=USERNAME,
        project_name=get_project_name(debug=False),
        pattern=None
    )
    return [exp for exp in all_experiments if experiment_filter(exp)]


def get_parameter(experiment: APIExperiment, parameter_name: str) -> Optional[str]:
    """Gets a parameter from an APIExperiment."""
    summary: List[dict] = experiment.get_parameters_summary()
    for curr_param in summary:
        if curr_param["name"] == parameter_name:
            return curr_param["valueCurrent"]
    return None


def get_group_size(experiment: APIExperiment) -> int:
    """Gets the group size from an APIExperiment."""
    return int(get_parameter(experiment, "group_size"))


def get_score_stat(experiment: APIExperiment, stat: str) -> Dict[str, float]:
    score_stat = {}
    for curr_metric_name in metric_names:
        summary: List[dict] = experiment.get_metrics_summary()
        for curr_metric_value in filter(lambda x: x["name"] == f"general_{curr_metric_name}_{stat}", summary):
            score_stat[curr_metric_name] = float(curr_metric_value["valueCurrent"])
    return score_stat


def plot_data(data: Dict[int, Dict[str, float]], stat: str) -> None:
    """Adds a scatter plot to the panel.
    makes sure that:
    the y scale is from 0 to 1
    the graph is added to the panel using ui.display_figure
    data format:
    {group_size: {metric_name: score}}
    """
    curr_figure = plt.gcf()
    curr_figure.clear()
    plt.figure(figsize=FIGURE_SIZE)
    colors = ["red", "green", "blue"]
    for curr_metric_name, curr_color in zip(metric_names, colors):
        plt.scatter(
            data.keys(),
            [data[curr_group_size][curr_metric_name] for curr_group_size in data.keys()],
            color=curr_color,
            label=curr_metric_name,
        )
    plt.legend()
    plt.xlabel(X_LABEL)
    y_label = f"BERT scores {stat}"
    plt.ylabel(y_label)
    plt.title(f"{y_label} as a function of {X_LABEL}")
    plt.ylim(0, 1)
    plt.show()
    plt.clf()


def generate_plot(stat_name: str) -> None:
    display(Markdown(f"## {stat_name}"))
    group_size_to_score_stats: Dict[int, Dict[str, float]] = {}
    for exp in get_relevant_experiments():
        group_size: int = get_group_size(exp)
        curr_exp_stats: Dict[str, float] = get_score_stat(exp, stat_name)
        if len(curr_exp_stats) > 0:
            group_size_to_score_stats[group_size] = curr_exp_stats
    if len(group_size_to_score_stats) > 0:
        plot_data(group_size_to_score_stats, stat_name)
    else:
        display(Markdown("No experiments with this stat"))


def get_duration(exp: APIExperiment) -> float:
    """Returns the duration of an experiment in hours"""
    duration_milliseconds = exp.get_metadata()["durationMillis"]
    return duration_milliseconds / (1000*60*60)


def plot_duration():
    group_size_to_duration = {get_group_size(exp): get_duration(exp) for exp in get_relevant_experiments()}
    plt.autoscale(True)
    curr_figure = plt.gcf()
    curr_figure.clear()
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(
        group_size_to_duration.keys(),
        group_size_to_duration.values(),
    )
    plt.xlabel(X_LABEL)
    y_label = "experiment duration in hours"
    plt.ylabel(y_label)
    plt.title(f"{y_label} as a function of {X_LABEL}")
    plt.show()
    plt.clf()
