import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets as data

import squamish.main
import fri

import pickle
import dataclasses
from typing import List

import pathlib
PATH = pathlib.Path(__file__).parent 
TMP = PATH / ("./output/tmp")
EXP_FILE = "NL_experiment_results.pickle"

@dataclasses.dataclass
class Result:
    dataset : str
    model : str
    features : list
    score : float
    params : dict

@dataclasses.dataclass
class Experiment:
    results : List[Result] = dataclasses.field(default_factory=list)
    #def __init__(self):
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

    

def run_experiment():
    state = np.random.RandomState(123)
    n_jobs = 1
    repeats = 5

    # Models
    f = fri.FRI(
        fri.ProblemName.CLASSIFICATION,
        n_probe_features=50,
        verbose=False,
        random_state=state,
        n_jobs=n_jobs,
    )
    sq = squamish.main.Main(random_state=state, n_jobs=n_jobs)
    models = {"FRI": f, "Sq": sq}

    # Data Generation
    generate_func = data.make_classification
    default_params = {
        "n_samples": 300,
        "n_classes": 2,
        "n_clusters_per_class": 2,
        "class_sep": 0.5,
        "hypercube": True,
        "shift": 0.0,
        "scale": 1.0,
        "shuffle": False,
        "random_state": state,
    }
    datasets = {
        1: {"n_features": 20, "n_informative": 10, "n_redundant": 0,},
        2: {"n_features": 20, "n_informative": 5, "n_redundant": 5,},
        #3: {"n_features": 20, "n_informative": 5, "n_repeated": 10,},
    }

    exp = Experiment()
    for d_name, d_param in datasets.items():
        # Generate data with parameters
        cur_param = dict(default_params,**d_param)
        X, y = generate_func(**cur_param)
        # Run models
        for m_name, model in models.items():
            for r in range(repeats):
                model.fit(X, y)
                score = model.score(X, y)
                features = model.relevance_classes_
                result = Result(d_name, m_name, features, score, cur_param)
                exp.add(result)
        #print(exp)

    with open(TMP/EXP_FILE, "wb") as file:
        pickle.dump(exp,file)

    return exp

def analyze(exp=None):
    if exp is None:
        try:
            with open(TMP/EXP_FILE, "rb") as file:
                exp = pickle.load(file)
        except IOError:
            exp = run_experiment()
    
if __name__ == "__main__":
    exp = run_experiment()

