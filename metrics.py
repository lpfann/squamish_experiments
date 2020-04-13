from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def get_truth(params):
    strong = params["strong"]
    weak = params["weak"]
    irrel = params["irr"]
    truth = [True] * (strong + weak) + [False] * irrel
    return truth

def get_truth_new(params):
    try:
        strong = params["n_strel"]
    except KeyError:
        strong = 0
    try:
        repeated = params["n_repeated"]
    except KeyError:
        repeated = 0
    try:
        weak = params["n_redundant"]
    except KeyError:
        weak = 0
    irrel = params["n_features"] - strong - weak - repeated
    truth = [True] * (strong + weak + repeated) + [False] * irrel
    return truth

def get_truthAR(params):
    strong=params["n_strel"]
    weak=params["n_redundant"]
    irrel = params["n_features"] - strong - weak
    truth = [2] * strong + [1]*weak + [0] * irrel
    return truth
    
# We only take truth if relevance type (1 or 2) 
def get_truth_onetype(params, reltype):
    truth = get_truthAR(params)
    truth = np.array(truth)
    if reltype==0:
        raise Exception("0 makes no sense")
    return truth == reltype

def get_scores_for_set(truth, fset, scores=[precision_score,recall_score,f1_score], AR=True):
    if AR:
        if 2 in fset:
            fset = np.array(fset > 0).astype(int)
    out = []
    for scf in scores:
        score = scf(truth,fset)
        out.append(score)
    return out