from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN
from collections import Counter
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample, check_X_y, check_random_state,shuffle
import math
import scipy.io as sio
from pathlib import Path
from fri import genClassificationData
import numpy as np
import os

class Dataset(object):
    def __init__(
        self,
        X,
        y,
        missingvals=True,
        balance=True,
        standardize=True,
        random_state=None,
        test_size=0.2,
        names=None,
    ):
        self.X, self.y = check_X_y(X, y, force_all_finite=False)
        y = LabelEncoder().fit_transform(y)
        if names is None and isinstance(
            X, pd.DataFrame
        ):  # Use column names if input is DataFrame
            names = X.columns.values
        else:
            names = np.array(names)
        self.names = names

        self.random_state = check_random_state(random_state)
        self.bootstraps = None
        self.cvfolds = None

        self.test_size = test_size
        self.balance = balance
        self.missingvals = missingvals
        self.standardize = standardize

    def get_simple_train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        if self.missingvals:
            # Impute missing vals with column mean
            imp = SimpleImputer()
            imp.fit(X_train)
            X_train = imp.transform(X_train)
            X_test = imp.transform(X_test)

        if self.balance:
            # Balance out classes
            # Not needed when we use frequency binning!
            balancer = SMOTEENN(random_state=self.random_state)
            X_train, y_train = balancer.fit_resample(X_train, y_train)

        if self.standardize:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    def get_bootstraps(self, n_bootstraps=20, perc=0.8):
        X, y, *_ = self.get_simple_train_test_split()
        n = X[0].shape[0]
        bs = [
            resample(
                X,y,
                replace=False,
                n_samples=math.floor(perc * n),
                random_state=self.random_state
            )
            for bs in range(n_bootstraps)
        ]
        return bs

    def get_train_cv_folds(self, folds=10):
        X, y, *_ = self.get_simple_train_test_split()

        kfold = StratifiedKFold(n_splits=folds, random_state=self.random_state)
        folds = []
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            folds.append((X_train, y_train, X_test, y_test))
        return folds

    def get_all_cv_folds(self, folds=10):
        X, y = self.X, self.y

        kfold = ShuffleSplit(n_splits=folds, test_size=self.test_size, random_state=self.random_state)
        folds = []
        for train_index, test_index in kfold.split(X, y):
            # Split
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Standardize
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            folds.append((X_train, y_train, X_test, y_test))

        return folds


class Toyset(Dataset):
    def __init__(
        self, n_features=30, n_strel=4, n_redundant=4, n_samples=500, random_state=123, noise=0
    ):
        self.n_features = n_features
        self.n_strel = n_strel
        self.n_redundant = n_redundant
        self.n_samples = n_samples
        self.random_state = random_state
        self.noise = noise

        X, y = genClassificationData(
            n_features=n_features,
            n_strel=n_strel,
            n_redundant=n_redundant,
            n_samples=n_samples,
            random_state=random_state,
            noise=noise
        )

        super().__init__(
            X, y, balance=False, missingvals=False, random_state=random_state
        )

    def get_bootstraps(self, n_bootstraps=20, perc=0.8):
        # Generate new data according to prototype set used to create object
        bs = []
        for i in range(n_bootstraps):
            X, y = genClassificationData(
                n_features=self.n_features,
                n_strel=self.n_strel,
                n_redundant=self.n_redundant,
                n_samples=self.n_samples,
                random_state=self.random_state,
                noise=self.noise
            )
            y = LabelEncoder().fit_transform(y)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            bs.append((X, y))
        self.bootstraps = bs
        return self.bootstraps



def import_Fibrosis(random_state, **kwargs):
    NA_THRESH = 0.92
    table = pd.read_csv("../data/Fibrosis.csv")

    d = table.shape[1]
    table = table.dropna(thresh=NA_THRESH * d)  # Drop samples with > 90% NaN

    # Our predictor variable
    Y_raw = table.Fibrosis
    X_raw = table.drop("Fibrosis", 1)

    dataset = Dataset(X_raw, Y_raw, random_state=random_state, **kwargs)
    return dataset


def import_colposcopy(random_state, **kwargs):
    NA_THRESH = 0.92
    table = pd.read_csv("../data/Quality Assessment - Digital Colposcopy/green.csv")

    d = table.shape[1]
    table = table.dropna(thresh=NA_THRESH * d)  # Drop samples with > 90% NaN
    table = table[table.columns[~table.columns.str.contains("expert")]]
    # Our predictor variable
    Y_raw = table.consensus
    X_raw = table.drop("consensus", 1)

    dataset = Dataset(X_raw, Y_raw, random_state=random_state, **kwargs)
    return dataset

def import_cervical(random_state, **kwargs):
    NA_THRESH = 0.92

    ds = pd.read_csv("../data/risk_factors_cervical_cancer.csv")
    y = ds.Schiller

    X = ds.iloc[:,:-4]
    X[X=="?"]=np.nan
    #X = X.fillna(0)

    dataset = Dataset(X, y, random_state=random_state, **kwargs)
    return dataset

def import_FLIP(random_state, **kwargs):
    NA_THRESH = 0.6
    path = "../data/FLIP.csv"
    joined = os.path.join(os.path.dirname(__file__), path)
    table = pd.read_csv(joined)

    d = table.shape[1]
    table = table.dropna(thresh=NA_THRESH * d)  # Drop samples with > 90% NaN

    # Our predictor variable
    Y_raw = table.FLIP_k
    no_class = Y_raw[Y_raw.isnull()].index
    Y_raw = Y_raw.drop(no_class)

    X_raw = table.drop("FLIP_k", 1)
    X_raw = X_raw.drop(no_class)

    dataset = Dataset(X_raw, Y_raw, random_state=random_state, **kwargs)
    return dataset


def import_wbc(random_state, **kwargs):
    dset = pd.read_csv(
        "../data/wdbc.data", index_col=0, header=None
    )

    y = LabelEncoder().fit_transform(dset[1]).flatten()

    X = dset.drop(1, axis=1)

    dataset = Dataset(X, y, random_state=random_state, **kwargs)
    return dataset


def import_T21(random_state, **kwargs):
    random_state = check_random_state(random_state)
    NA_THRESH = 0.9
    # cache big file
    file = Path("../data/T21.feather")
    if file.exists():
        table = pd.read_feather(str(file))
    else:
        table = pd.read_excel(
            io="../data/data_50K_raw_N_T21.xlsx"
        )
        table.to_feather(str(file))

    d = table.shape[1]
    table = table.dropna(thresh=NA_THRESH * d)  # Drop samples with > 90% NaN

    # Our predictor variable
    Y_raw = table.Class
    Y_raw = Y_raw.replace("a-T21", 1)
    Y_raw = Y_raw.replace("Normal", 0)

    X = table.drop("Class", 1)
    X = X.drop(X.columns[19:], 1)
    X = X.drop("ID Sort", 1)

    # Downsample maj. class (50k) to 1k
    majority_class = Y_raw[Y_raw == 0].index
    state = random_state
    keep = 1000
    n_majority = len(majority_class) - keep
    drop_index = state.choice(majority_class, size=n_majority, replace=False)
    X = X.drop(drop_index)
    Y_raw = Y_raw.drop(drop_index)

    dataset = Dataset(X, Y_raw, random_state=random_state, **kwargs)
    return dataset


def import_SPECTF(random_state, **kwargs):
    random_state = check_random_state(random_state)
    NA_THRESH = 0.9

    table = pd.read_csv("../data/SPECTF.csv")

    d = table.shape[1]
    table = table.dropna(thresh=NA_THRESH * d)  # Drop samples with > 90% NaN

    # Our predictor variable
    Y_raw = table["0"]
    no_class = Y_raw[Y_raw.isnull()].index
    Y_raw = Y_raw.drop(no_class)

    X_raw = table.drop(table.columns[0], axis=1)
    X_raw = X_raw.drop(X_raw.columns[0], axis=1)
    names = [
        "F1R",
        "F1S",
        "F2R",
        "F2S",
        "F3R",
        "F3S",
        "F4R",
        "F4S",
        "F5R",
        "F5S",
        "F6R",
        "F6S",
        "F7R",
        "F7S",
        "F8R",
        "F8S",
        "F9R",
        "F9S",
        "F10R",
        "F10S",
        "F11R",
        "F11S",
        "F12R",
        "F12S",
        "F13R",
        "F13S",
        "F14R",
        "F14S",
        "F15R",
        "F15S",
        "F16R",
        "F16S",
        "F17R",
        "F17S",
        "F18R",
        "F18S",
        "F19R",
        "F19S",
        "F20R",
        "F20S",
        "F21R",
        "F21S",
        "F22R",
        "F22S",
    ]

    names = np.array(names)

    dataset = Dataset(X_raw, Y_raw, random_state=random_state, names=names, **kwargs)
    return dataset


def get_datasets(seed):
    print("FIBROSIS")
    set_fibrosis = import_Fibrosis(seed)
    print("FLIP")
    set_flip = import_FLIP(seed)
    print("T21")
    set_t21 = import_T21(seed)
    print("SPECTF")
    set_spectf = import_SPECTF(seed)
    print("WBC")
    set_WBC = import_wbc(seed)
    print("colposcopy")
    set_col = import_colposcopy(seed)
    print("cervical")
    set_cervical = import_cervical(seed)

    datasets = {
        "fibrosis": set_fibrosis,
        #"colposcopy": set_col,
        #"cervical": set_cervical,
        #"flip": set_flip,
        #"t21": set_t21,
        #"spectf": set_spectf,
        #"wbc": set_WBC,
    }
    return datasets



toy_set_params = {
            "Set1": {"n": 150, "strong": 6, "weak": 0, "irr": 6},
            "Set2": {"n": 150, "strong": 0, "weak": 6, "irr": 6},
            "Set3": {"n": 150, "strong": 3, "weak": 4, "irr": 3},
            "Set4": {"n": 256, "strong": 6, "weak": 6, "irr": 6},
            "Set5": {"n": 512, "strong": 1, "weak": 2, "irr": 11},
            "Set6": {"n": 200, "strong": 1, "weak": 20, "irr": 0},
            "Set7": {"n": 200, "strong": 1, "weak": 20, "irr": 20},
        }
#toy_set_params = {
#            "Set8": {"n": 10000, "strong": 10, "weak": 20, "irr": 10},
#            "Set9": {"n": 10000, "strong": 10, "weak": 20, "irr": 200},
#        }

def get_toy_datasets(seed, toy_set_params=toy_set_params, noise=0.0):
    datasets = {}
    for name, params in toy_set_params.items():
        n_strel = params["strong"]
        n_redundant = params["weak"]
        n_irrel = params["irr"]
        n_features = n_strel + n_redundant + n_irrel
        n_samples = params["n"]
        datasets[name] = Toyset(
            n_features=n_features,
            n_strel=n_strel,
            n_redundant=n_redundant,
            n_samples=n_samples,
            random_state=seed,
            noise=noise
        )
    return datasets


if __name__ == "__main__":
    toydatasets = get_toy_datasets(123)
    realdatasets = get_datasets(123)
    print(toydatasets, realdatasets)
