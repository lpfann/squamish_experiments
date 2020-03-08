import time
from collections import defaultdict
import dill as pickle
import joblib
from numpy.random import RandomState
from dask_jobqueue import SGECluster
from dask.distributed import Client
import dask
from job import Job
import fsmodel
import import_data
import argparse

## functions which are run in the worker threads for parallel computation
def worker():
    return job.run_esann_test()


def experiment(parallel, models, datasets):
    sets = []
    for modelname, model in models.items():
        for setname, dataset in datasets.items():
            bootstraps = dataset.bootstraps
            for X, y in bootstraps:
                bs = X, y
                job = Job(
                    traindata=bs,
                    model=models[modelname],
                    modelname=modelname,
                    setname=setname,
                )
                sets.append(job)

    results = []
    # Using pool map to run map in paralell
    # We also use tqdm to display progress bar
    results = parallel(map(joblib.delayed(worker_stability), sets))
    if len(results) < 1:
        raise Exception()

    return results
