import abc
import sys
import warnings

import numpy as np
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state

import fri
import linear_models

import squamish

class FSmodel(object):
    """
    Abstract class for all models which are used for feature selection
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, random_state=None):
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
    def predict(self,X):
        return


class LM(FSmodel):
    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        # if type == "l1":
        #     l1_ratio = 1
        # if type == "l2":
        #     l1_ratio = 0
        # if type == "elasticnet":
        #     l1_ratio = 0.5


    def fit(self, X, Y):
        model = linear_models.RegularizedLinearOrdinalRegression()
        tuned_parameters = {"C":  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "l1_ratio":[0, 0.01, 0.1, 0.2, 0.5, 0.7,1]}
        cv = 3
        gridsearch = GridSearchCV(
            model, tuned_parameters, cv=cv, verbose=0, error_score=np.nan,n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gridsearch.fit(X, Y)
        self.model = gridsearch.best_estimator_
        self.selector = RFECV(estimator=self.model, cv=5, min_features_to_select=2)
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
    def __init__(self, probtype="ordreg",random_state=None):
        super().__init__(random_state=random_state)

        self.model = fri.FRI(fri.ProblemName.CLASSIFICATION,
            random_state=self.random_state,
            slack_regularization=0.1,
            slack_loss=0.1,
            n_probe_features=50,
            n_jobs=7,
            n_param_search=50
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
    def __init__(self,random_state=None):
        super().__init__(random_state=random_state)

        self.model = squamish.Main(
            random_state=self.random_state
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

        return self

    def support(self):
        return self.model.support_
        
    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)

class AllFeatures(FSmodel):
    def __init__(self, random_state=None):
        super().__init__()

    def fit(self, X, Y):
        self.d = X.shape[1]
        pass

    def support(self):
        return np.ones(self.d)

    def score(self, X, y):
        return 0


def get_models(seed):
    # FRI
    fri_model_exc = FRI(random_state=seed)
    # L1 LM
    #l1lm = LM(random_state=seed, type="l1")
    # ElasticNet
    eelm = LM(random_state=seed)

    sq = SQ(random_state=seed)
    # Ridge
    #ridge = LM(random_state=seed, type="l2")
    # Dummy selector
    #afm = AllFeatures()

    models = {
        #"FRI_exc": fri_model_exc,
        #"FRI_imp": fri_model_imp,
        #"Lasso": l1lm,
        "ElasticNet": eelm,
        "SQ": sq,
        #"Ridge" : ridge,
        #"AllFeatures": afm,
    }
    return models

if __name__ == "__main__":
    m = get_models(123)
    print(m)