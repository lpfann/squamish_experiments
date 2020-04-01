from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit,StratifiedKFold
from sklearn.utils import resample, check_X_y, check_random_state
from sklearn.utils import shuffle
import math
import scipy.io as sio
import pathlib
from pathlib import Path
import numpy as np
from fri import genOrdinalRegressionData
from sklearn.impute import SimpleImputer
import re

class Dataset(object):
    def __init__(
        self,
        X,
        y,
        missingvals=True,
        balance=False, # We dont need balancing if we do frequency binning from regression data which is balanced by definition
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

        X, y = genOrdinalRegressionData(
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
            X, y = genOrdinalRegressionData(
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

def get_datasets(seed):

    datasets = {
        # "flip": set_flip,
        # "t21": set_t21,
        # "spectf": set_spectf,
        # "wbc": set_WBC,
    }
    datasets = import_ordinal_sets_with_folds()

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


def freq_binning(X_reg, y_reg, random_state, n_bins=10):

    n, d = X_reg.shape
    bin_size = int(np.floor(n / n_bins))
    rest = int(n - (bin_size * n_bins))

    # Sort the target values and rearange the data accordingly
    sort_indices = np.argsort(y_reg)
    X = X_reg[sort_indices]
    y = y_reg[sort_indices]

    # Assign ordinal classes as target values
    for i in range(n_bins):
        if i < rest:
            y[(bin_size + 1) * i:] = i
        else:
            y[(bin_size * i) + rest:] = i

    X, y = shuffle(X, y, random_state=random_state)

    return X, y


def import_train_test_splits(path):

    path = pathlib.Path(path)
    n = len(list(path.glob("train_*.*")))

    splits = [0]*n

    for filepath in sorted(path.glob("train_*.*")):
        s = filepath
        split = re.compile(r'(\d+)$').search(str(s)).group(1)
        
        data = np.loadtxt(s)
        X = data[:,:-1]
        y = data[:,-1]
        y = y-1
        splits[int(split)] = {"train_X":X, "train_y":y}

    for filepath in sorted(path.glob("test_*.*")):
        s = filepath
        split = re.compile(r'(\d+)$').search(str(s)).group(1)
        
        data = np.loadtxt(s)
        X = data[:,:-1]
        y = data[:,-1]
        y = y-1
        splits[int(split)].update({"test_X":X, "test_y":y})

    return list(splits)


def import_ordinal_sets_with_folds():

        data_paths = {
        "Contact-lenses": {"path": '../../data/ordinal-classification-datasets/contact-lenses/gpor'},
        "Eucalyptus": {"path": '../../data/ordinal-classification-datasets/eucalyptus/gpor'},
        "Newthyroid": {"path": '../../data/ordinal-classification-datasets/newthyroid/gpor'},
        "Pasture": {"path": '../../data/ordinal-classification-datasets/pasture/gpor'},
        "Squash-stored": {"path": '../../data/ordinal-classification-datasets/squash-stored/gpor'},
        "Squash-unstored": {"path": '../../data/ordinal-classification-datasets/squash-unstored/gpor'},
        "TAE": {"path": '../../data/ordinal-classification-datasets/tae/gpor'},
        "Winequality-red": {"path": '../../data/ordinal-classification-datasets/winequality-red/gpor'},
        "Bondrate": {"path": '../../data/ordinal-classification-datasets/bondrate/gpor'},
        "Automobile": {"path": '../../data/ordinal-classification-datasets/automobile/gpor'},
        }
        datasets = {}
        for name, params in data_paths.items():
            path = params["path"]
            splits = import_train_test_splits(path)
            for split in splits:
                X_train = split["train_X"]
                X_test = split["test_X"]
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                split["train_X"] = X_train
                split["test_X"] = X_test

            datasets[name] = splits

        return datasets

def import_benchmark_data(random_state, **kwargs):

    #
    #
    # regression sets with frequency binning
    #################
    data_paths_1 = {
        "Pyrimidines": {"path": '../../data/Pyrimidines/pyrim.data', "delimiter": ',', "test": 24, "d": 27},
        "MachineCPU": {"path": '../../data/MachineCPU/machine.data', "delimiter": ',', "test": 59, "d": 6},
        "Boston": {"path": '../../data/Boston/housing.data', "delimiter": ',', "test": 206, "d": 13},
        "Computer": {"path": '../../data/Computer/cpu_act.data', "delimiter": ',', "test": 4182, "d": 21},
        "California": {"path": '../../data/California/cal_housing.data', "delimiter": ',', "test": 15640, "d": 8},
    }
    data_paths_2 = {
        "Bank": {"path": '../../data/Bank/bank32full.data', "test": 5182, "d": 32},
        "Census": {"path": '../../data/Census/census-house/house-price-8L/Prototask.data', "test": 16784, "d": 8},
    }

    #
    #
    # Ordinal regression Sets
    ##########
    data_paths_3 = {
        "Contact-lenses": {"path": '../../data/ordinal-classification-datasets/contact-lenses/gpor/test_contact-lenses.0',
                           "path2": '../../data/ordinal-classification-datasets/contact-lenses/gpor/train_contact-lenses.0',
                           "test": 0.25, "d": 6},
        "Eucalyptus": {"path": '../../data/ordinal-classification-datasets/eucalyptus/gpor/test_eucalyptus.0',
                       "path2": '../../data/ordinal-classification-datasets/eucalyptus/gpor/train_eucalyptus.0',
                       "test": 0.25, "d": 91},
        "Newthyroid": {"path": '../../data/ordinal-classification-datasets/newthyroid/gpor/test_newthyroid.0',
                       "path2": '../../data/ordinal-classification-datasets/newthyroid/gpor/train_newthyroid.0',
                       "test": 0.25, "d": 5},
        "Pasture": {"path": '../../data/ordinal-classification-datasets/pasture/gpor/test_pasture.0',
                    "path2": '../../data/ordinal-classification-datasets/pasture/gpor/train_pasture.0',
                    "test": 0.25, "d": 5},
        "Squash-stored": {"path": '../../data/ordinal-classification-datasets/squash-stored/gpor/test_squash-stored.0',
                          "path2": '../../data/ordinal-classification-datasets/squash-stored/gpor/train_squash-stored.0',
                          "test": 0.25, "d": 51},
        "Squash-unstored": {"path": '../../data/ordinal-classification-datasets/squash-unstored/gpor/test_squash-unstored.0',
                            "path2": '../../data/ordinal-classification-datasets/squash-unstored/gpor/train_squash-unstored.0',
                            "test": 0.25, "d": 52},
        "TAE": {"path": '../../data/ordinal-classification-datasets/tae/gpor/test_tae.0',
                "path2": '../../data/ordinal-classification-datasets/tae/gpor/train_tae.0',
                "test": 0.25, "d": 54},
        "Winequality-red": {"path": '../../data/ordinal-classification-datasets/winequality-red/gpor/test_winequality-red.0',
                            "path2": '../../data/ordinal-classification-datasets/winequality-red/gpor/train_winequality-red.0',
                            "test": 0.25, "d": 11}
    }

    data_paths_4 = {
        "Bondrate": {"path": '../../data/ordinal-classification-datasets/bondrate/gpor/test_bondrate.0',
                     "path2": '../../data/ordinal-classification-datasets/bondrate/gpor/train_bondrate.0',
                     "test": 0.25, "d": 37}
    }

    data_paths_5 = {
        "Automobile": {"path": '../../data/ordinal-classification-datasets/automobile/gpor/test_automobile.0',
                       "path2": '../../data/ordinal-classification-datasets/automobile/gpor/train_automobile.0',
                       "test": 0.25, "d": 71},
    }

    datasets = {}

    FOLDS = 25
    # for name, params in data_paths_1.items():
    #     data = np.loadtxt(params["path"], delimiter=params["delimiter"])
    #     X, y = freq_binning(data[:,0:params["d"]], data[:,-1], random_state)
    #     dataset = Dataset(X,y,random_state=random_state, test_size=params["test"], **kwargs)
    #     repeated_folds = dataset.get_all_cv_folds(folds=FOLDS)
    #     dataset.folds = repeated_folds
    #     datasets[name] = dataset

    # for name, params in data_paths_2.items():
    #     data = np.loadtxt(params["path"])
    #     X, y = freq_binning(data[:, 0:params["d"]], data[:, -1], random_state)
    #     dataset = Dataset(X, y, random_state=random_state, test_size=params["test"], **kwargs)
    #     repeated_folds = dataset.get_all_cv_folds(folds=FOLDS)
    #     dataset.folds = repeated_folds
    #     datasets[name] = dataset

    FOLDS = 35

    for name, params in data_paths_3.items():
        data1 = np.loadtxt(params["path"])
        data2 = np.loadtxt(params["path2"])
        data = np.append(data1, data2, axis=0)
        X = data[:,0:params["d"]]
        y = data[:,-1]
        y = y - 1  # classes should start at 0
        dataset = Dataset(X, y, random_state=random_state, test_size=params["test"], **kwargs)
        repeated_folds = dataset.get_all_cv_folds(folds=FOLDS)
        dataset.folds = repeated_folds
        datasets[name] = dataset

    for name, params in data_paths_4.items():
        data1 = np.loadtxt(params["path"])
        data2 = np.loadtxt(params["path2"])
        data = np.append(data1, data2, axis=0)
        X = data[:,0:params["d"]]
        y = data[:,-1]
        y = y - 1  # classes should start at 0
        #double point no. 14 to have two members in his class
        pointX = X[14].reshape(1, params["d"])
        pointy = y[14]
        X = np.append(X, pointX, axis=0)
        y = np.append(y, pointy)
        dataset = Dataset(X, y, random_state=random_state, test_size=params["test"], **kwargs)
        repeated_folds = dataset.get_all_cv_folds(folds=FOLDS)
        dataset.folds = repeated_folds
        datasets[name] = dataset

    for name, params in data_paths_5.items():
        data1 = np.loadtxt(params["path"])
        data2 = np.loadtxt(params["path2"])
        data = np.append(data1, data2, axis=0)
        X = data[:,0:params["d"]]
        y = data[:,-1]
        y = y - 1  # classes should start at 0
        #double every point to avoid empty classes in further calculation
        #X = np.append(X, X, axis=0)
        #y = np.append(y, y)
        dataset = Dataset(X, y, random_state=random_state, test_size=params["test"], **kwargs)
        repeated_folds = dataset.get_all_cv_folds(folds=FOLDS)
        dataset.folds = repeated_folds
        datasets[name] = dataset

    return datasets



if __name__ == "__main__":
    #datasets_1 = get_toy_datasets(123)
    #datasets_2 = import_benchmark_data(123)
    datasets_3 = import_ordinal_sets_with_folds()
    #cvs = datasets["toy_red_sparse"].get_bootstraps()
    #print(len(cvs))

