from contextlib import contextmanager
import sys, os
import logging
logging = logging.getLogger("Job")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


import sklearn.linear_model
import sklearn.model_selection

import time
from collections import namedtuple

import numpy as np

Result_Stability = namedtuple(
    "Result_Stability", ["modelname", "setname", "featset", "runtime", "score"]
)
Result_Performance = namedtuple(
    "Result_Performance",
    [
        "modelname",
        "setname",
        "trainScore",
        "testScore",
        "traindec",
        "trainpredict",
        "trainy",
        "testdec",
        "testpredict",
        "testy",
    ],
)


class Job(object):

    """Job class which gets run in worker threads so we can run the experiments in parallel.

        
        Attributes:
            data array: Dataset in X,y tuple
            featset array: selected features for the model which is getting tested
            model objet: model which is getting tested
            modelname str: the name 
            result_performance object: object which gets returned from worker thread to main thread. It saves the result data from the performance test.
            result_stability object: see above, for stability test
            score float: score of model on training set
            setname str: Name of the dataset
        """

    def __init__(
        self, traindata=None, testdata=None, model=None, modelname=None, setname=None
    ):
        self.setname = setname
        self.modelname = modelname
        self.traindata = traindata
        self.testdata = testdata
        self.model = model

    def run_esann_test(self):
        model = self.model
        print("Running {} on set {}".format(self.modelname, self.setname))

        score = ordinal_scores
        # Get data
        traindata, testdata = self.traindata, self.testdata

        X, y = traindata
        # Timing the model
        start_time = time.time()
        try:
            with suppress_stdout():
                model.ft(X, y)  # Run the model

        except Exception as e:
            print("Error at set {} with model {}".format(self.setname, self.modelname))
            print("X shape is {}, y classes are {}".format(X.shape, np.unique(y)))
            print(e)
            return None

        delta_time = time.time() - start_time

        # Retrieve the selected features
        selected_features = model.support()

        train_scores = {}
        testpredict = model.predict(X)
        for score_type in ["mze", "mae", "mmae"]:
            testscore = score(testpredict, y, score_type, return_error=True)
            train_scores[score_type] = testscore

        X, y = testdata
        test_scores = {}
        testpredict = model.predict(X)
        for score_type in ["mze", "mae", "mmae"]:
            testscore = score(testpredict, y, score_type, return_error=True)
            test_scores[score_type] = testscore

        results = {}
        results["train_scores"] = train_scores
        results["test_scores"] = test_scores
        results["features"] = selected_features
        results["runtime"] = delta_time
        results["setname"] = self.setname
        results["modelname"] = self.modelname

        return results

    def run_one_bootstrap(self):
        """Method which runs the model on the set and saves the resulting feature set for later analysis.
                We also record runtime and training error.
            """
        data, model = self.traindata, self.model
        logging.info("Running {} on set {}".format(self.modelname, self.setname))

        X, y = data
        # Timing the model
        start_time = time.time()
        model.fit(X, y)  # Run the model
        #try:
        #    print()
        #    #with suppress_stdout():
        #except Exception as e:
        #    logging.exception("Error at set {} with model {}".format(self.setname, self.modelname))
        #    logging.exception("X shape is {}, y classes are {}".format(X.shape, np.unique(y)))
        #    return None

        delta_time = time.time() - start_time
        # Retrieve the selected features
        selected_features = model.support()
        self.featset = selected_features
        # Save trainig score
        self.score = model.score(X, y)
        # Build result object which the main thread can get back

        results = {}
        results["train_scores"] = self.score
        results["features"] = selected_features
        results["runtime"] = delta_time
        results["setname"] = self.setname
        results["modelname"] = self.modelname

        # Remove model so we do not have to pickle the fitted model, we dont need it  and avoid error with rpy models
        del self.model
        self.model = None

        return results

    def run_one_perf_test(self):
        """Method to run a neutral benchmark on a uniform model with the previous selected feature set.
            We want to see which feature set is the best or has the most information.

            
            """
        # Get data
        traindata, testdata = self.trainData, self.testData

        X, y = traindata
        # Check if feature set as more than one feature
        if np.array(self.featset).ndim > 1:
            self.featset = self.featset[0]
        # reduce our data set to the selected features
        X = np.compress(self.featset, X, axis=1)

        # Neutral model
        model = sklearn.linear_model.LogisticRegression(
            multi_class="multinomial", solver="newton-cg", fit_intercept=True
        )
        tuned_parameters = [{"C": np.logspace(-6, 4, 11)}]
        cv = 5

        # We use gridsearch to find good parameters
        gridsearch = sklearn.model_selection.GridSearchCV(
            model, tuned_parameters, scoring=None, n_jobs=1, cv=cv, iid=True
        )
        gridsearch.fit(X, y)
        est = gridsearch.best_estimator_

        # Record scores
        trainScore = est.score(X, y)
        traindec = est.decision_function(X)
        trainpredict = est.predict(X)
        trainy = y
        # Scores on the testset
        X_test, y_test = testdata
        X_test = np.compress(self.featset, X_test, axis=1)

        testpredict = est.predict(X_test)

        testscore = est.score(X_test, y_test)


        # We save the decision function, we can calculate ROC curves later in the analysis
        # TODO: do we need the decision function?, are there alternatives for ord. Regr?
        testdec = est.decision_function(X_test)
        testy = y_test

        result = Result_Performance(
            modelname=self.modelname,
            setname=self.setname,
            trainScore=trainScore,
            testScore=testscore,
            traindec=traindec,
            trainpredict=trainpredict,
            trainy=trainy,
            testdec=testdec,
            testpredict=testpredict,
            testy=testy,
        )
        self.result_performance = result

        return self

