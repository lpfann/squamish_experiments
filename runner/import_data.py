from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTEENN
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.utils import resample, check_X_y, check_random_state
import math
from fri import genClassificationData
import numpy as np
import pathlib

RELATIVE_PATH = pathlib.Path(__file__).parent.resolve()

import logging

logging = logging.getLogger(__name__)


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
                X,
                y,
                replace=False,
                n_samples=math.floor(perc * n),
                random_state=self.random_state,
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

        kfold = ShuffleSplit(
            n_splits=folds, test_size=self.test_size, random_state=self.random_state
        )
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
        self,
        n_features=30,
        n_strel=4,
        n_redundant=4,
        n_samples=500,
        random_state=123,
        noise=0,
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
            noise=noise,
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
                noise=self.noise,
            )
            y = LabelEncoder().fit_transform(y)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            bs.append((X, y))
        self.bootstraps = bs
        return self.bootstraps