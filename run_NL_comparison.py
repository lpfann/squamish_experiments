import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets as data
import pandas as pd

import squamish.main
import fri

import pickle
import dataclasses
from typing import List

import pathlib

PATH = pathlib.Path(__file__).parent
TMP = PATH / ("./tmp")
EXP_FILE = "NL_experiment_results"

import metrics


@dataclasses.dataclass
class Result:
    dataset: str
    model: str
    features: list
    precision: float
    recall: float
    f1: float
    train_score: float
    params: dict


@dataclasses.dataclass
class Experiment:
    results: List[Result] = dataclasses.field(default_factory=list)
    # def __init__(self):
    #    self.results : list(Result) = []

    def res_by_set(self, dataset):
        for r in self.results:
            if r.dataset == dataset:
                yield r

    def res_by_model(self, model):
        for r in self.results:
            if r.model == model:
                yield r

    def add(self, result):
        self.results.append(result)


def run_experiment():
    state = np.random.RandomState(123)
    n_jobs = 1
    repeats = 2

    # Models
    f = fri.FRI(
        fri.ProblemName.CLASSIFICATION,
        n_probe_features=50,
        verbose=False,
        random_state=state,
        n_jobs=n_jobs,
    )
    sq = squamish.main.Main(random_state=state, n_jobs=n_jobs, verbose=False)
    models = {"FRI": f, "Sq": sq}

    # Data Generation
    generate_func = data.make_classification
    default_params = {
        "n_samples": 1000,
        "n_classes": 2,
        "n_clusters_per_class": 2,
        "class_sep": 0.5,
        "hypercube": True,
        "shift": 0.0,
        "scale": 1.0,
        "shuffle": False,
        "random_state": state,
    }
    datasets = {
        1: {"n_features": 20, "n_informative": 10, "n_redundant": 0,},
        2: {"n_features": 20, "n_informative": 5, "n_redundant": 5,},
        3: {"n_features": 20, "n_informative": 5, "n_repeated": 10,},
        4: {"n_features": 100, "n_informative": 20, "n_redundant": 20},
    }
    res_list = []
    for d_name, d_param in datasets.items():
        # Generate data with parameters
        cur_param = dict(default_params, **d_param)
        X, y = generate_func(**cur_param)
        # Run models
        for m_name, model in models.items():
            for r in range(repeats):
                model.fit(X, y)
                train_score = model.score(X, y)
                features = model.relevance_classes_
                truth = metrics.get_truth_new(cur_param)
                scores = metrics.get_scores_for_set(truth, features)
                result = Result(
                    d_name, m_name, features, *scores, train_score, cur_param
                )
                res_list.append(vars(result))
    exp = pd.DataFrame(res_list)
    exp.to_csv(TMP / (EXP_FILE + ".csv"))
    exp.to_pickle(TMP / (EXP_FILE + ".pickle"))

    return exp


def analyze(exp=None):
    if exp is None:
        try:
            exp = pd.read_pickle(TMP / (EXP_FILE + ".pickle"))
        except NameError:
            exp = run_experiment()


if __name__ == "__main__":
    exp = run_experiment()
