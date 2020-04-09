import abc
import sys
import warnings

import numpy as np
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.linear_model import (
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
    SGDClassifier,
)
from sklearn.model_selection import cross_val_score

import sklearn.feature_selection as fs
import lightgbm

import fri

import squamish
import logging


class FSmodel(object):
    """
    Abstract class for all models which are used for feature selection
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, random_state=None, n_jobs=1):
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.support_ = None
        self.model = None

    @abc.abstractmethod
    def fit(self, X, Y):
        return

    @abc.abstractmethod
    def score(self, X, y):
        return

    @abc.abstractmethod
    def support(self):
        return self.support_

    @abc.abstractmethod
    def predict(self, X):
        return


class LM(FSmodel):
    def __init__(self, random_state=None, n_jobs=1):
        super().__init__(random_state=random_state, n_jobs= n_jobs)

    def fit(self, X, Y):
        model = SGDClassifier(max_iter=100, tol=1e-3, random_state=self.random_state)
        tuned_parameters = {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0, 0.01, 0.1, 0.2, 0.5, 0.7, 1],
        }
        cv = 3
        gridsearch = GridSearchCV(
            model, tuned_parameters, cv=cv, verbose=0, error_score=np.nan, n_jobs=self.n_jobs
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gridsearch.fit(X, Y)
        self.model = gridsearch.best_estimator_
        self.selector = RFECV(
            estimator=self.model, cv=5, min_features_to_select=2, n_jobs=self.n_jobs
        )
        self.selector.fit(X, Y)
        self.support_ = self.selector.get_support()
        return self

    def support(self):
        return self.support_

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)


class FRI(FSmodel):
    def __init__(self, random_state=None, n_jobs=1):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.model = fri.FRI(
            fri.ProblemName.CLASSIFICATION,
            random_state=self.random_state,
            w_l1_slack=0.1,
            loss_slack=0.1,
            n_probe_features=50,
            n_jobs=self.n_jobs,
            n_param_search=50,
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

        return self

    def support(self):
        return self.model.relevance_classes_

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.optim_model_.predict(X)


class SQ(FSmodel):
    def __init__(self, random_state=None, n_jobs=1):
        super().__init__(random_state=random_state, n_jobs=n_jobs)

        self.model = squamish.Main(random_state=self.random_state, n_jobs=self.n_jobs)

    def fit(self, X, Y):

        self.model.fit(X, Y)
        logging.info(self.score(X, Y))

        return self

    def support(self):
        return self.model.relevance_classes_

    def score(self, X, y):
        self.model.rfmodel.fit(X,y)
        return self.model.rfmodel.score(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RF(FSmodel):
    def __init__(self, random_state=None, n_jobs=1, cv=5):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        PARAMS = {
            "num_leaves": 32,
            "max_depth": 5,
            "boosting_type": "rf",
            "bagging_fraction": 0.632,
            "bagging_freq": 1,
            # "feature_fraction": 0.8,
            # Add nulled parameters to surpress warning messages
            "subsample": None,
            "subsample_freq": None,
            "colsample_bytree": None,
            "verbose": -1,
        }
        self.treemodel = lightgbm.LGBMClassifier(
            random_state=self.random_state.randint(10000), n_jobs=self.n_jobs, **PARAMS
        )
        self.model = fs.RFECV(self.treemodel, cv=cv)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        logging.info(self.score(X, Y))
        return self

    def support(self):
        return self.model.support_

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)


class AllFeatures(FSmodel):
    def __init__(self, random_state=None, n_jobs=1):
        super().__init__()

    def fit(self, X, Y):
        self.d = X.shape[1]
        pass

    def support(self):
        return np.ones(self.d)

    def score(self, X, y):
        return 0
