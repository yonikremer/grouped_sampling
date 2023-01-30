"""This is a script that plots the results of the experiments and saves the figures to the plots folder"""
from os.path import abspath, dirname, join
from typing import Dict, List, Optional

from comet_ml import API, APIExperiment
from matplotlib import pyplot as plt
from pandas import DataFrame

from evaluation import (
    BERT_SCORES,
    STAT_NAME_TO_FUNC,
    WORKSPACE,
    get_comet_api_key,
    get_project_name,
)

api = API(api_key=get_comet_api_key())
stat_names = tuple(stat_name for stat_name, _ in STAT_NAME_TO_FUNC)
metric_names = BERT_SCORES
X_LABEL = "group size"
curr_dir = dirname(abspath(__file__))
PLOTS_FOLDER = join(curr_dir, "plots", "third_part")
TABLES_FOLDER = join(curr_dir, "tables", "third_part")

plt.autoscale(False)


def experiment_filter(exp: APIExperiment) -> bool:
    # return true if the experiment have a tag that says "third part"
    return "third part" in exp.get_tags()


def get_relevant_experiments() -> List[APIExperiment]:
    all_experiments: List[APIExperiment] = api.get_experiments(
        workspace=WORKSPACE,
        project_name=get_project_name(debug=False),
        pattern=None)
    unsorted_relevant_experiments = (exp for exp in all_experiments
                                     if experiment_filter(exp))
    return sorted(unsorted_relevant_experiments, key=get_duration)


def get_parameter(experiment: APIExperiment,
                  parameter_name: str) -> Optional[str]:
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
        for curr_metric_value in filter(
                lambda x: x["name"] == f"general_{curr_metric_name}_{stat}",
                summary):
            score_stat[curr_metric_name] = float(
                curr_metric_value["valueCurrent"])
    return score_stat


def save_plot_from_data(data: Dict[int, Dict[str, float]], stat: str) -> None:
    """Adds a scatter plot to the panel.
    makes sure that:
    the y scale is from 0 to 1
    the graph is added to the panel using ui.display_figure
    data format:
    {group_size: {metric_name: score}}
    """
    curr_figure = plt.gcf()
    curr_figure.clear()
    colors = ["red", "green", "blue"]
    for curr_metric_name, curr_color in zip(metric_names, colors):
        plt.scatter(
            data.keys(),
            [
                data[curr_group_size][curr_metric_name]
                for curr_group_size in data.keys()
            ],
            color=curr_color,
            label=curr_metric_name,
        )
    plt.xscale("log")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(X_LABEL)
    y_label = f"BERT scores {stat}"
    plt.ylabel(y_label)
    title = f"{y_label} as a function of {X_LABEL}"
    plt.title(title)
    fig_path = join(PLOTS_FOLDER, f"{title}.png")
    plt.savefig(
        fig_path,
        bbox_inches="tight",
        pad_inches=0,
    )
    print(f"Saved a plot in {fig_path}")
    plt.clf()


def save_stat_plot(stat_name: str) -> None:
    group_size_to_score_stats: Dict[int, Dict[str, float]] = {}
    for exp in get_relevant_experiments():
        group_size: int = get_group_size(exp)
        curr_exp_stats: Dict[str, float] = get_score_stat(exp, stat_name)
        if len(curr_exp_stats) > 0:
            group_size_to_score_stats[group_size] = curr_exp_stats
    if len(group_size_to_score_stats) > 0:
        save_plot_from_data(group_size_to_score_stats, stat_name)
    else:
        raise RuntimeError(
            f"Could not find any experiments with the stat {stat_name}.")


def get_duration(exp: APIExperiment) -> float:
    """Returns the duration of an experiment in hours"""
    duration_milliseconds = exp.get_metadata()["durationMillis"]
    return duration_milliseconds / (1000 * 60 * 60)


def save_duration_plot():
    group_size_to_duration = {
        get_group_size(exp): get_duration(exp)
        for exp in get_relevant_experiments()
    }
    plt.autoscale(True)
    curr_figure = plt.gcf()
    curr_figure.clear()
    plt.scatter(
        group_size_to_duration.keys(),
        group_size_to_duration.values(),
    )
    plt.xscale("log")
    plt.xlabel(X_LABEL)
    y_label = "experiment duration in hours"
    plt.ylabel(y_label)
    title = f"{y_label} as a function of {X_LABEL}"
    plt.title(title)
    fig_path = (join(PLOTS_FOLDER, join(PLOTS_FOLDER, f"{title}.png")), )
    plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved a plot in {fig_path}")
    plt.clf()


def save_stat_table(stat_name: str) -> None:
    group_size_to_score_stats: Dict[int, Dict[str, float]] = {}
    for exp in get_relevant_experiments():
        group_size: int = get_group_size(exp)
        curr_exp_stats: Dict[str, float] = get_score_stat(exp, stat_name)
        if len(curr_exp_stats) > 0:
            group_size_to_score_stats[group_size] = curr_exp_stats
    if len(group_size_to_score_stats) <= 0:
        raise RuntimeError(
            f"Could not find any experiments with the stat {stat_name}.")
    df = DataFrame()
    for curr_metric_name in metric_names:
        for curr_group_size in group_size_to_score_stats:
            df.loc[curr_group_size,
                   curr_metric_name] = group_size_to_score_stats[
                       curr_group_size][curr_metric_name]
    df = df.sort_index()
    df = df.round(3)
    df.to_csv(join(TABLES_FOLDER, f"{stat_name}.csv"))


if __name__ == "__main__":
    print("started")
    for curr_stat_name in stat_names:
        save_stat_plot(curr_stat_name)
        save_stat_table(curr_stat_name)
    save_duration_plot()
    print("Done")
