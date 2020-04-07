import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets as data
import arfs_gen
import pandas as pd
import argparse
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

from utils import print_df_astable

import sys

sys.path.append("./runner/")
import fsmodel, experiment_pipeline

# Data Generation
generate_func = arfs_gen.genClassificationData
default_params = {
    "n_samples": 1000,
    "linear":False
}
datasets = {
    "NL 1": {"n_features": 20, "n_strel": 10, "n_redundant": 0,},
    "NL 2": {"n_features": 20, "n_strel": 0, "n_redundant": 10,},
    "NL 3": {"n_features": 20, "n_strel": 5, "n_repeated": 5,},
    "NL 4": {"n_features": 100, "n_strel": 20, "n_redundant": 20},
}

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


def run_experiment(state = np.random.RandomState(123), n_jobs = -1,   repeats = 10):

    models = experiment_pipeline.get_models(state,n_jobs=n_jobs)

    default_params["random_state"] = state
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
                features = model.support()
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


def run(recompute=True, n_jobs=-1):
    if not recompute:
        try:
            exp = pd.read_pickle(TMP / (EXP_FILE + ".pickle"))
            return exp
        except NameError:
            pass
        except FileNotFoundError:
            pass
    exp = run_experiment()
    return exp


def analyze(exp=None):
    exp = exp.set_index(["dataset", "model"])

    # Take mean
    table = exp.groupby(["dataset", "model"]).mean()
    # Round
    table = table.round(decimals=2)
    # Reformat
    table = table.stack().swaplevel(i=-3, j=-1).unstack(["model"])
    print_df_astable(table, "per_dataset", folder="NL_toy_benchmarks")

    # Take mean over all data
    overallmean = exp.groupby(["model"]).mean()
    # Round
    overallmean = overallmean.round(decimals=2)
    print_df_astable(overallmean, "mean_stats", folder="NL_toy_benchmarks")

    preset = pd.DataFrame(datasets).T
    preset[preset.isna()] = 0
    preset = preset.astype(int)
    preset.index.names = ["Set"]
    print_df_astable(preset.T, "presets", folder="NL_toy_benchmarks")
    return table, overallmean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", type=bool, default=False)
    parser.add_argument("--jobs", type=int, default=-1)

    args = parser.parse_args()

    exp = run(args.recompute, args.jobs)
    table, overallmean = analyze(exp)
    
