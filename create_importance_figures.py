#!/usr/bin/env python
# coding: utf-8

import pathlib

from sklearn.utils import check_random_state
import os

PATH = pathlib.Path("./output/figures/importance_plots")
os.makedirs(PATH, exist_ok=True)
import lightgbm
import boruta
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.model_selection import ParameterGrid
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)
matplotlib.rcParams["pgf.rcfonts"] = False
# Load style file
plt.style.use("PaperDoubleFig.mplstyle")

def get_fs(estimator):
    fset = fs.SelectFromModel(
        prefit=True, estimator=estimator, threshold="mean"
    ).get_support()
    return fset


def model(random_state=None, params=None):
    return lightgbm.LGBMClassifier(random_state=random_state, **params)


def train_and_get_imps(X, y, random_state, params=None):
    rf = model(random_state=random_state, params=params)
    rf.fit(X, y)

    return rf.feature_importances_




def plot_imp_list(imp_list, line=True, color=None):
    fig, ax1 = plt.subplots()
    frame = pd.DataFrame(imp_list)
    frame.plot(kind="box", ax=ax1, color=color)
    if line:
        for q in frame.mean(1).quantile([0.1, 0.5, 0.9]):
            plt.axhline(q)


def plot_class_list(imp_list):
    fig, ax1 = plt.subplots()
    frame = pd.DataFrame(imp_list).mean()
    frame.plot(kind="bar", ax=ax1)


def get_relev_class_RFE(X, y, random_state=None, params=None):
    rfc = fs.RFECV(model(random_state=random_state, params=params), cv=5)
    rfc.fit(X, y)
    return rfc.support_.astype(int)


from arfs_gen import genClassificationData


def data(informative=5, redundant=10, d=17, n=300, state= np.random.RandomState(seed=1231241)):

    X, y = genClassificationData(
        n_features=d,
        n_redundant=redundant,
        n_strel=informative,
        n_samples=n,
        random_state=state,
    )
    X = scale(X)
    return X, y


def benchmark(best_params_rf, X, y, random_state, prefix="" ):

    imp_list = [
        train_and_get_imps(
            X, y, random_state=random_state.randint(1e6), params=best_params_rf
        )
        for t in range(10)
    ]

    plot_imp_list(imp_list)
    plt.title("Distribution of importance values")
    plt.savefig(PATH / (prefix + "distimps.pdf"))
    rel_class_list = [
        get_relev_class_RFE(
            X, y, random_state=random_state.randint(1e6), params=best_params_rf
        )
        for t in range(10)
    ]
    plot_class_list(rel_class_list)
    plt.title("Frequency of inclusion in Minimal Feature Set (RFECV)")
    plt.savefig(PATH / (prefix + "freqimpsMR.pdf"))


def borutabench(best_params_rf, X, y, random_state, prefix=""):
    bor = boruta.BorutaPy(model(params=best_params_rf,random_state=random_state))
    bor.fit(X, y)
    plot_class_list([bor.support_])
    plt.title("Frequency of inclusion in AllRel Set (Boruta)")
    plt.savefig(PATH / (prefix + "freqimpsAR.pdf"))


def bench_and_plot(random_state):
    random_state = check_random_state(random_state)
    # # Gain Importance

    # ## 1.0 Feature Fraction

    best_params_rf = {
        "max_depth": [5],
        "boosting_type": ["rf"],
        "bagging_fraction": [0.632],
        "bagging_freq": [1],
        "feature_fraction": [1],
        "importance_type": ["gain"],
    }
    X, y = data()
    best_params_rf = ParameterGrid(best_params_rf)[0]
    benchmark(best_params_rf, X, y,  random_state,prefix="1ff")
    borutabench(best_params_rf, X, y, random_state, prefix="1ff")

    # # ## 0.5 Feature Fraction
    #
    # best_params_rf = {
    #     "max_depth": [5],
    #     "boosting_type": ["rf"],
    #     "bagging_fraction": [0.632],
    #     "bagging_freq": [1],
    #     "feature_fraction": [0.5],
    #     "importance_type": ["gain"],
    # }
    # X, y = data()
    # best_params_rf = ParameterGrid(best_params_rf)[0]
    # benchmark(best_params_rf, X, y,  random_state,prefix="05ff")
    # borutabench(best_params_rf, X, y, random_state, prefix="05ff")

    # # 0.1 Feature Fraction

    best_params_rf = {
        "max_depth": [5],
        "boosting_type": ["rf"],
        "bagging_fraction": [0.632],
        "bagging_freq": [1],
        "feature_fraction": [0.1],
        "importance_type": ["gain"],
    }
    X, y = data()
    best_params_rf = ParameterGrid(best_params_rf)[0]
    name = "01ff"
    benchmark(best_params_rf, X, y,  random_state,prefix=name)
    borutabench(best_params_rf, X, y, random_state, prefix=name)

if __name__ == '__main__':
    bench_and_plot(1337)