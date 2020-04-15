import dataclasses
import pathlib
import sys
from typing import List

import arfs_gen
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import scale

sys.path.append("./runner/")
import experiment_pipeline
import metrics
from utils import print_df_astable
import click

PATH = pathlib.Path(__file__).parent
TMP = PATH / ("./tmp")
EXP_FILE = "NL_experiment_results"


# Data Generation
generate_func = arfs_gen.genClassificationData
default_params = {"n_samples": 1000, "linear": False}
datasets = {
    "NL 1": {"n_features": 20, "n_strel": 10, "n_redundant": 0,},
    "NL 2": {"n_features": 20, "n_strel": 4, "n_redundant": 10,},
    "NL 3": {"n_features": 50, "n_strel": 10, "n_redundant": 10,},
    "NL 4": {"n_features": 80, "n_strel": 10, "n_redundant": 10},
}

ARmodels = ["FRI", "SQ"]


@dataclasses.dataclass
class Result:
    dataset: str
    model: str
    features: list
    precision: float
    recall: float
    f1: float
    accuracy: float
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


def run_experiment(state=np.random.RandomState(123), n_jobs=-1, repeats=2):

    models = experiment_pipeline.get_models(state, n_jobs=n_jobs)

    default_params["random_state"] = state
    res_list = []
    for d_name, d_param in datasets.items():
        print("*" * 20)
        print(d_name)
        # Generate data with parameters
        cur_param = dict(default_params, **d_param)
        X, y = generate_func(**cur_param)
        X = scale(X)

        # Run models
        for m_name, model in models.items():
            print("*" * 20)
            print(m_name)
            for r in range(repeats):
                print("*" * 10)
                print(r)
                model.fit(X, y)
                # train_score = model.score(X, y)
                # Cast -1,1 encoding to 0,1 encoding of classes (in case of fri)
                accuracy = model.score(X, y)
                features = model.support()
                truth = metrics.get_truth_new(cur_param)
                scores = metrics.get_scores_for_set(truth, features)
                result = Result(d_name, m_name, features, *scores, accuracy, cur_param)
                res_list.append(vars(result))
    exp = pd.DataFrame(res_list)
    exp.to_csv(TMP / (EXP_FILE + ".csv"))
    exp.to_pickle(TMP / (EXP_FILE + ".pickle"))

    return exp


def get_Accuracy_per_Relevance_Class(table):
    onlyAR = table[table.model.isin(ARmodels)]
    onlyAR = onlyAR.set_index(["dataset", "model"])
    onlyAR_features = onlyAR[["features"]]

    def rowfunc(row, reltype, scorefunc):
        # Functhon which applies score function on feature list and compares with ground truth
        data = row.name[0]
        features = row["features"]
        pred = features == reltype

        truth = metrics.get_truth_onetype(datasets[data], reltype)
        if sum(truth) == 0:
            return np.nan
        score = scorefunc(truth, pred, zero_division=1)

        # if score == 0:
        #    score = np.nan
        return score

    weakly_precision = onlyAR_features.apply(rowfunc, axis=1, args=[1, precision_score])
    weakly_recall = onlyAR_features.apply(rowfunc, axis=1, args=[1, recall_score])
    weakly = pd.concat([weakly_precision, weakly_recall], axis=1).rename(
        columns={0: "precision", 1: "recall"}
    )

    strongly_precision = onlyAR_features.apply(
        rowfunc, axis=1, args=[2, precision_score]
    )
    strongly_recall = onlyAR_features.apply(rowfunc, axis=1, args=[2, recall_score])
    strongly = pd.concat([strongly_precision, strongly_recall], axis=1).rename(
        columns={0: "precision", 1: "recall"}
    )

    combined = pd.concat([weakly, strongly], axis=1, keys=["Weakly", "Strongly"]).round(
        decimals=2
    )

    mean_datamodel = combined.groupby(["dataset", "model"]).mean()

    mean_model = mean_datamodel.groupby(["model"]).mean()

    return mean_datamodel, mean_model


def analyze(exp=None):

    #
    #
    # Accuracy over all features
    #
    exp_with_index = exp.set_index(["dataset", "model"])
    # Take mean
    table = exp_with_index.groupby(["dataset", "model"]).mean()
    # Round
    table = table.round(decimals=2)
    # Reformat
    table = table.stack().swaplevel(i=-3, j=-1).unstack(["model"])
    print_df_astable(table, "per_dataset", folder="NL_toy_benchmarks")

    # Take mean over all data
    overallmean = exp_with_index.groupby(["model"]).mean()
    # Round
    overallmean = overallmean.round(decimals=2)
    print_df_astable(overallmean, "mean_stats", folder="NL_toy_benchmarks")

    preset = pd.DataFrame(datasets).T
    preset[preset.isna()] = 0
    preset = preset.astype(int)
    preset.index.names = ["Set"]
    print_df_astable(preset.T, "presets", folder="NL_toy_benchmarks")

    #
    #
    # Analyse Feature Classes Accuracy
    #
    fclasses_combined, fclasses_meanmodel = get_Accuracy_per_Relevance_Class(exp)
    print_df_astable(
        fclasses_combined, "accuracy_per_fclass", folder="NL_toy_benchmarks", na_rep="-"
    )
    print_df_astable(
        fclasses_meanmodel, "accuracy_per_fclass_mean", folder="NL_toy_benchmarks"
    )


@click.command()
@click.option("--recompute", default=False, is_flag=True)
@click.option("--n_jobs", default=-1, show_default=True)
@click.option("--seed", default=123, show_default=True)
@click.option("--repeats", default=2, show_default=True)
def run(recompute, n_jobs, seed, repeats):
    exp = None
    if not recompute:
        try:
            exp = pd.read_pickle(TMP / (EXP_FILE + ".pickle"))
        except NameError:
            pass
        except FileNotFoundError:
            pass
    if exp is None:
        exp = run_experiment(
            state=np.random.RandomState(seed), n_jobs=n_jobs, repeats=repeats
        )
    analyze(exp)


if __name__ == "__main__":
    run()
