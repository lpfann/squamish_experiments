import dill as pickle
import numpy as np
import pandas as pd
import argparse
import sys

from numpy.random import RandomState
from sklearn.metrics import precision_score, recall_score, f1_score

import experiment_pipeline

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pgf import FigureCanvasPgf

matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)
matplotlib.rcParams["pgf.rcfonts"] = False
# Load style file
plt.style.use("PaperDoubleFig.mplstyle")

toy_set_params = experiment_pipeline.toy_set_params

parser = argparse.ArgumentParser(description="Start experiment manually from CLI")
parser.add_argument("--iters", type=int, default=25)
parser.add_argument("--resfile", type=str)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()


def load_file(name):
    with open("./results/{}".format(name), "rb") as f:
        stability_res = pickle.load(f)
        #accuracy_res = pickle.load(f)
    return stability_res, accuracy_res


if len(sys.argv) < 1:
    seed = RandomState(1337)
    n_bs = 3
    experiment_pipeline.main_exp(
        n_bootstraps=n_bs,
        SEED=seed,
        filename="paper_experiment",
        toy=True,
    )
    stability_res, accuracy_res = load_file("res_paper_experiment.dat")
else:
    if args.resfile is not None:
        with open("./results/{}".format(args.resfile), "rb") as f:
            stability_res = pickle.load(f)
            #accuracy_res = pickle.load(f)
        n_bs = args.iters
    else:
        seed = RandomState(args.seed)
        n_bs = args.iters
        experiment_pipeline.main_exp(
            n_bootstraps=n_bs,
            SEED=seed,
            filename="paper_experiment",
            toy=True,
        )
        stability_res, accuracy_res = load_file("res_paper_experiment.dat")

datasets = set(list(zip(*stability_res.keys()))[0])
models = set(list(zip(*stability_res.keys()))[1])

mod_list = list(models)
model_sorter = {
    "FRI": 0,
    "SQ":1,
    "RF":2,
    "ElasticNet": 3,
    "AllFeatures": 4,
    "Lasso": 5,
    "Ridge": 6
}
mod_list = sorted(mod_list, key=lambda x: model_sorter[x])

short_model_names = {
    "ElasticNet": "EN",
    "FRI": "FRI",
    "Lasso": "Lasso",
    "AllFeatures": "AF",
    "Boruta": "Boruta",
    "EFS": "EFS",
    "Ridge": "Ridge",
    "RF": "RF",
    "SQ": "SQ",
}

sim_set_names = {
    "Set1": "Set 1",
    "Set2": "Set 2",
    "Set3": "Set 3",
    "Set4": "Set 4",
    "Set5": "Set 5",
    "Set6": "Set 6",
    "Set7": "Set 7",
    "Set8": "Set 8",
    "Set9": "Set 9"
    }

# Set consistent color pallete
pal = dict(zip(models, sns.color_palette("Set1", n_colors=len(models))))

renamed_sets = []
for name in datasets:
    if "Set" in name:
        renamed_sets.append(sim_set_names[name])
    else:
        renamed_sets.append(name)
# dataset_list = [sim_set_names[name] for name in datasets]
datasorter = {
    "Set 1": 1,
    "Set 2": 2,
    "Set 3": 3,
    "Set 4": 4,
    "Set 5": 5,
    "Set 6": 6,
    "Set 7": 7,
    "Set 8": 7,
    "Set 9": 7,
    "Automobile": 8,
    "Contact-lenses": 9,
    "Eucalyptus": 10,
    "Newthyroid": 11,
    "Pasture": 12,
    "Squash-stored": 13,
    "Squash-unstored": 14,
    "TAE": 15,
    "Winequality-red": 16,
    "Bondrate": 17,
    "Pyrimidines": 18,
    "MachineCPU": 19,
    "Bank": 20,
    "Boston": 21,
    "Computer": 22,
    "California": 23,
    "Census": 24

}
dataset_list = sorted(renamed_sets, key=lambda x: datasorter[x])

# Convert result dictionaries to dataframe
index = pd.MultiIndex.from_tuples(stability_res.keys())
list_df = pd.DataFrame(
    [pd.Series(value) for value in stability_res.values()], index=index
)
stability_res = list_df.dropna().T  # Drop invalid results

#index = pd.MultiIndex.from_tuples(accuracy_res.keys())
#list_df = pd.DataFrame(
#    [pd.Series(value) for value in accuracy_res.values()], index=index
#)
#accuracy_res = list_df.dropna().T  # Drop invalid results


def print_df_astable(df, filename=None):
    output = df.to_latex(multicolumn=False, bold_rows=True)
    if filename is not None:
        with open("../output/tables/{}.tex".format(filename), "w") as f:
            f.write(output)
    return output


def get_sim_param_table(sets):
    sim_params = (
        pd.DataFrame.from_dict(toy_set_params)
        .T.rename(index=sim_set_names)
    )
    #sim_params.columns = ["Samples","Strongly Relevant", "Weakly Relevant","Irrelevant"]
    sim_params.index.name = "Set"
    return sim_params



def stability_plot(stability_res, pal, order):
    # Using Nogueiras stability measure
    stabf = (
        pd.DataFrame(stability_res)
        .applymap(lambda x: x["features"].astype(int))
        .apply(lambda array: st.getStability(list(array)))
    )
    # stabf = pd.DataFrame(stability_res).applymap(lambda x: x.featset).apply(stability)
    # Nur relevant feature stability measure -:
    # stabf = pd.DataFrame(stability_res).applymap(lambda x: x.featset[:sum(get_toy_set_params(x.setname)[1:3])]).apply(stability)
    stabf = stabf.reset_index().rename(
        columns={"level_0": "data", "level_1": "model", 0: "stability"}
    )
    # stabf.sort_values(by="data").groupby(["model", "data"]).mean().unstack("data")
    stab_sorted = stabf.sort_values(by="data")
    stab_sorted = stab_sorted[stab_sorted["model"] != "AllFeatures"]
    stab_sorted.groupby(["model", "data"]).mean().unstack("data")
    stab_sorted.data = stab_sorted.data.apply(
        lambda name: sim_set_names[name] if name in sim_set_names else name
    )

    #sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    # fig.set_figwidth(3.5)
    fig.set_figheight(2.1)

    plot_stab_real = sns.barplot(
        x="data",
        y="stability",
        hue="model",
        data=stab_sorted,
        palette=pal,
        order=order,
        saturation=0.8,
    )
    plot_stab_real.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plot_stab_real.axvline(4.5,color="black",linewidth=3)
    plot_stab_real.set_yticks([0.25, 0.5, 0.75, 1])
    #plot_stab_real.grid(b=True, axis="y")
    plot_stab_real.set_ylim([0, 1])
    plt.savefig("./figures/stability_toy.pdf", bbox_inches="tight")
    # plt.savefig("../../../tex_bmc_paper/figures/stability_toy.pgf",bbox_inches="tight")


def get_truth(params):
    strong=params["strong"]
    weak=params["weak"]
    irrel=params["irr"]
    truth = [True] * (strong + weak) + [False] * irrel
    return truth


def selection_rate_plot(models, datasets, toy=True):
    def selection_rate(stability_res, model, dataset):
        res = stability_res[(dataset, model)]
        d = len(res[0]["features"])
        n_bs = len(res)
        tmp = []
        for r in res:
            tmp.append(r["features"])
        table = pd.DataFrame(tmp, index=np.arange(len(tmp)), columns=range(d))
        return table

    def get_relevance_class(params):
        strong=params["strong"]
        weak=params["weak"]
        irrel=params["irr"]
        truth = (
            [0] * strong
            + [1] * weak
            + [2] * irrel
        )
        return truth

    sorted_models_tmp = sorted(models)
    sorted_data_tmp = sorted(datasets, reverse=True)
    if toy:
        sorted_data_tmp = list(filter(lambda name: "Set" in name, sorted_data_tmp))
    else:
        sorted_data_tmp = list(filter(lambda name: "Set" not in name, sorted_data_tmp))
    print(sorted_data_tmp)
    # Setup grid of axes to plot into
    f, axarr = plt.subplots(
        nrows=len(sorted_models_tmp),
        ncols=len(sorted_data_tmp),
        sharex="col",
        sharey="row",
        squeeze=True,
    )
    # f.set_figheight(3)
    f.set_figwidth(5)
    # f.tight_layout()

    # Create legend with colored patches
    color_palette_3 = sns.color_palette(palette="colorblind", n_colors=3)
    relevance_classes = ["Strongly relevant", "Weakly relevant", "Irrelevant"]
    patches = []
    for i, rc in enumerate(relevance_classes):
        patch = mpatches.Patch(color=color_palette_3[i], label=rc)
        patches.append(patch)

    # Iterate over all model and data combinations and calculate the selectionrate using all experimental iterations
    for mi, m in zip(range(len(sorted_models_tmp)), sorted_models_tmp):
        for di, d in zip(range(len(sorted_data_tmp)), sorted_data_tmp):
            if toy:
                # Get color according to relevance class
                colors = get_relevance_class(toy_set_params[d])
                colors = [color_palette_3[c] for c in colors]

            # Calculate rate
            rates = selection_rate(stability_res, m, d).mean()
            axis = axarr[mi, di]
            index = np.arange(len(rates))
            # Plot Bars
            axis.bar(index, rates, color=colors if toy else None)
            axis.set_aspect("auto")
            axis.set_xticks([])
            axis.set_xlim([0, len(rates)])
            mean_set_size = selection_rate(stability_res, m, d).sum(1).mean()
            axis.set_xlabel(mean_set_size, fontsize=7)
            # Only label first elements in row/column
            axis.yaxis.set_tick_params(length=0)
            if di == 0:
                axis.set_ylabel(short_model_names[m])
                axis.set_yticklabels([0, 1], fontsize=7)
            if mi == 0:
                if toy:
                    axis.set_title(sim_set_names[d])
                else:
                    axis.set_title(d)

    # Add the legend
    if toy:
        plt.legend(
            handles=patches,
            bbox_to_anchor=(0.5, -0.1),
            bbox_transform=plt.gcf().transFigure,
            ncol=3,
            loc=8,
            fontsize=10,
            columnspacing=1,
        )
    f.savefig(
        "../output/autofigs/{}_fsfrequency.pdf".format("toy" if toy else "real"),
        bbox_inches="tight",
    )


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
        .rename(index=sim_set_names)
        .sort_index()
        .round(decimals=2)
    )
    return accuracy_on_sims


def get_sim_scores(stability_res):

    toyframe = stability_res.iloc[
        :, stability_res.columns.get_level_values(0).str.contains("Set")
    ]
    print(toyframe)
    def get_score_of_series(series, scorefnc):
        setname = series.name[0]

        def get_score(result):
            featset = result["features"]
            if 2 in featset:
                featset = np.array(featset>0).astype(int)
            truth_set = get_truth(toy_set_params[setname])
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

    #toy_scores = pd.concat([toy_precision, toy_recall, toy_f1])
    toy_scores = toy_f1

    # toy_f1.groupby(["model", "data"]).mean().unstack()

    grouped_toy_scores = (
        toy_scores.groupby(["model", "data", "type"]).mean().unstack(level="type")
    )
    # grouped_toy_scores = grouped_toy_scores.unstack("data")

    renamed_toy_scores = (
        grouped_toy_scores.round(decimals=2).unstack(1).rename(columns=sim_set_names)
    )
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
    runtime_frame = runtime_frame.rename(columns=sim_set_names).sort_index(axis=1).T
    return runtime_frame



sim_params = get_sim_param_table(list(datasets))
print_df_astable(sim_params, "sim_params")
print("#################### Simulation parameters")
print(sim_params)

sim_scores = get_sim_scores(stability_res)
print("#################### Simulation Scores")
print(print_df_astable(sim_scores, "sim_scores"))

sim_accuracy = get_sim_accuracy(stability_res)
print("#################### Training accuracy")
print(print_df_astable(sim_accuracy, "sim_accuracy"))

#stability_plot(stability_res, pal, dataset_list)

#selection_rate_plot(models, datasets)
#selection_rate_plot(models, datasets, toy=False)

# TODO: score ersetzen
#auc_frame_test, auc_mean_real = get_auc_table(accuracy_res)
#auc_mean_real = print_df_astable(auc_mean_real, "real_auc")
#print("#################### Mean of AUC")
#print(auc_mean_real)

#xi_squred_test(auc_frame_test)

runtime = get_runtime_table(stability_res)
runtime = print_df_astable(runtime, "runtime")
print("#################### Runtime of methods")
print(runtime)
