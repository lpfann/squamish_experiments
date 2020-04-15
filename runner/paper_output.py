import dill as pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

import experiment_pipeline

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from sklearn.utils import check_random_state

import pathlib

RES_PATH = pathlib.Path(__file__).parent / ("./results/")
OUTPUT_PATH = pathlib.Path(__file__).parent / ("../output/tables/toy_benchmarks/")
RELATIVE_PATH = pathlib.Path(__file__).parent.resolve()

import sys

sys.path.append("../")
from utils import print_df_astable


def load_file(path):
    with open(path, "rb") as f:
        stability_res = pickle.load(f)
    return stability_res


def _print_df_astable(df, filename=None):
    output = print_df_astable(df, filename, folder="toy_benchmarks")
    return output


def get_sim_param_table(toy_set_params):

    sim_params = pd.DataFrame.from_dict(toy_set_params).T
    sim_params.index.name = "Set"
    return sim_params


def get_truth(params):
    strong = params["strong"]
    weak = params["weak"]
    irrel = params["irr"]
    truth = [True] * (strong + weak) + [False] * irrel
    return truth


def get_sim_accuracy(stability_res):
    toy_accuracy = (
        pd.DataFrame(stability_res)
        .applymap(lambda r: r["train_scores"])
        .T.stack()
        .rename("accuracy")
    )

    accuracy_on_sims = (
        toy_accuracy.groupby(level=[0, 1])
        .mean()
        .unstack()
        .sort_index()
        .round(decimals=2)
    )
    return accuracy_on_sims


def get_sim_scores(stability_res, toy_set_params):

    toyframe = stability_res.iloc[
        :, stability_res.columns.get_level_values(0).str.contains("Set")
    ]
    print(toyframe)

    def get_score_of_series(series, scorefnc):
        setname = series.name[0]

        def get_score(result):
            featset = result["features"]
            if 2 in featset:
                featset = np.array(featset > 0).astype(int)
            try:
                # Try old naming of Sets withhout whitespace after "Set" (for  e.g. Set1)
                truth_set = get_truth(toy_set_params[setname])
            except KeyError:  # Error with new format e.g. "Set 1"
                newkey = setname[:3] + " " + setname[-1]
                truth_set = get_truth(toy_set_params[newkey])
            return scorefnc(truth_set, featset)

        prec_vec = map(get_score, series)
        return list(prec_vec)

    toy_precision = toyframe.apply(get_score_of_series, axis=0, args=[precision_score])
    toy_recall = toyframe.apply(get_score_of_series, axis=0, args=[recall_score])
    toy_f1 = toyframe.apply(get_score_of_series, axis=0, args=[f1_score])

    toy_precision = (
        toy_precision.T.stack()
        .reset_index()
        .drop("level_2", 1)
        .rename(columns={"level_0": "data", 0: "score", "level_1": "model"})
    )
    toy_precision["type"] = "precision"
    toy_recall = (
        toy_recall.T.stack()
        .reset_index()
        .drop("level_2", 1)
        .rename(columns={"level_0": "data", 0: "score", "level_1": "model"})
    )
    toy_recall["type"] = "recall"
    toy_f1 = (
        toy_f1.T.stack()
        .reset_index()
        .drop("level_2", 1)
        .rename(columns={"level_0": "data", 0: "score", "level_1": "model"})
    )
    toy_f1["type"] = "f1"

    # toy_scores = pd.concat([toy_precision, toy_recall, toy_f1])
    toy_scores = toy_f1

    # toy_f1.groupby(["model", "data"]).mean().unstack()

    grouped_toy_scores = (
        toy_scores.groupby(["model", "data", "type"]).mean().unstack(level="type")
    )
    # grouped_toy_scores = grouped_toy_scores.unstack("data")

    renamed_toy_scores = grouped_toy_scores.round(decimals=2).unstack(1)
    renamed_toy_scores = renamed_toy_scores.sort_index(axis=1).T

    return renamed_toy_scores


def get_runtime_table(res_dict):
    runtime_frame = pd.DataFrame(res_dict).applymap(lambda r: r["runtime"])
    runtime_frame = (
        runtime_frame.T.stack()
        .reset_index()
        .rename(columns={"level_0": "data", "level_1": "model", 0: "runtime"})
        .drop("level_2", axis=1)
    )
    runtime_frame = (
        runtime_frame.groupby(["model", "data"]).mean().unstack().astype(int)
    )
    runtime_frame = runtime_frame.sort_index(axis=1).T
    return runtime_frame


def run_new(n_bs=3, seed=1337):
    seed = check_random_state(seed)
    res = experiment_pipeline.main_exp(
        n_bootstraps=n_bs, SEED=seed, filename="paper_experiment", toy=True,
    )
    return res


def rename_old_namingscheme(stability_res):
    # If no whitespace between Set and Number rename keys to match new format WITH whitespace
    keys = stability_res.keys()
    probe = list(keys)[0]
    newstab = {}
    if " " not in probe[0]:
        for k in stability_res.keys():
            dataset, model = k
            newkey = dataset[:3] + " " + dataset[-1]
            newstab[(newkey, model)] = stability_res[k]
    return newstab


if __name__ == "__main__":
    matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)
    matplotlib.rcParams["pgf.rcfonts"] = False
    # Load style file
    style_file = (
        pathlib.Path(__file__).parent.parent.resolve() / "PaperDoubleFig.mplstyle"
    )
    plt.style.use(str(style_file))

    toy_set_params = experiment_pipeline.toy_set_params

    parser = argparse.ArgumentParser(description="Start experiment manually from CLI")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--resfile", type=str)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if len(sys.argv) < 1:
        # Run default, new experiment
        stability_res = run_new()
    else:
        if args.resfile is not None:
            # Load existing result
            path = pathlib.Path(args.resfile)
            print(f"Load from {path}")
            stability_res = load_file(path)
        else:
            stability_res = run_new(n_bs=args.iters, seed=args.seed)

    stability_res = rename_old_namingscheme(stability_res)

    # Convert result dictionaries to dataframe
    index = pd.MultiIndex.from_tuples(stability_res.keys())

    list_df = pd.DataFrame(
        [pd.Series(value) for value in stability_res.values()], index=index
    )
    stability_res = list_df.dropna().T  # Drop invalid results

    sim_params = get_sim_param_table(toy_set_params)
    _print_df_astable(sim_params, "sim_params")
    print("#################### Simulation parameters")
    print(sim_params)

    sim_scores = get_sim_scores(stability_res, toy_set_params)
    print("#################### Simulation Scores")
    print(_print_df_astable(sim_scores, "sim_scores"))

    sim_accuracy = get_sim_accuracy(stability_res)
    print("#################### Training accuracy")
    print(_print_df_astable(sim_accuracy, "sim_accuracy"))

    runtime = get_runtime_table(stability_res)
    runtime = _print_df_astable(runtime, "runtime")
    print("#################### Runtime of methods")
    print(runtime)
