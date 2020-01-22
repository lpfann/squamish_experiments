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

import logging
logger = logging.getLogger("Experiment")

## functions which are run in the worker threads for parallel computation
def worker_stability(job: Job):
    return job.run_one_bootstrap()
def worker_performance(job: Job):
    return job.run_one_perf_test()

## Main task delegation process  - only in main
def run_stability_experiment(models,datasets, parallel=None):
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
    if parallel is None:
        results = list(map(worker_stability,sets))
    else:
        results = parallel(map(joblib.delayed(worker_stability), sets))

    if len(results) < 1:
        raise Exception()

    return results

def run_performance_experiment(stabResults, parallel,datasets):
    """Delegator function which runs all stability tests
    
    Args:
        stabResults : Results from the stability experiment
        pool object: a multiprocessing Pool 
        datasets list: our datasets
    
    Returns:
        list: List of jobs with results
    """
    logger.debug(f"We started with:  {len(stabResults)}")

    jobs = []
    for job in stabResults:
        if job is None:
            continue
        name = job.setname
        data = datasets[name]
        train = data.X_train, data.y_train
        test = data.X_test, data.y_test
        job.trainData = train
        job.testData = test
        jobs.append(job)

    logger.debug(f"We have jobs: {len(jobs)}")
    # Using pool map to run map in paralell
    # We also use tqdm to display progress bar
    results = parallel(map(joblib.delayed(worker_performance), jobs))
    return results

def save_results(res_it, filename):
    end_time = time.time()
    if filename is not None:
        file = "res_{}.dat".format(filename)
    else:
        file = "res_{}.dat".format(end_time)

    with open("./results/{}".format(file), "wb") as f:
        res = []
        for res in res_it:
            res_dict = defaultdict(list)
            for result in res:
                if result is None:
                    continue
                else:
                    res_dict[(result["setname"], result["modelname"])].append(result)
            res.append(res_dict)
            pickle.dump(res_dict, f)
    return file, res

def get_bootstrapped_datasets(datasets, n_bootstraps):
    for name, ds in datasets.items():
        ds.bootstraps = ds.get_bootstraps(n_bootstraps=n_bootstraps, perc=1)
    return datasets
    
def main_exp(n_bootstraps=25, SEED=RandomState(1337), tempres=None, selectmodels=None, filename="test", threads=8, toy=True, noise=0, distributed=False):
    """Main function which gets data and models and starts experiments.
    We use a parallel worker model where individual models/set combinations are represented in Jobs which are run in worker threads.
    This function also uses two delegator functions to start the threads. (run_stability_experiment, run_performance_experiment)
    
    Args:
        n_bootstraps (int, optional): How many experiments per Dataset 
        SEED (TYPE, optional): Random seed
        tempres str: filename of temporary result, which we save when the computation stops in the second experiment 
        selectmodels list of str: Selected Models which we want to test. The model has to be existing in fsmodel.py 
        filename str: Output filename for result file.
    
    """
    logger.info("Start benchmark at {}".format(time.ctime()))

    # Get datasets and create bootstraps and folds in advance
    if toy:
        datasets = import_data.get_toy_datasets(SEED, noise=noise)
    else:
        datasets = import_data.get_datasets(SEED)
    #datasets.update(datasets_toy)

    #if toy:
    #    datasets = get_bootstrapped_datasets(datasets, n_bootstraps)

    datasets = get_bootstrapped_datasets(datasets, n_bootstraps)

    # Get models used in testing
    models = fsmodel.get_models(SEED)
    if selectmodels is not None:
        models = {k: v for k, v in models.items() if k in selectmodels}
    logger.info(f"models:{models}")

    # Cluster
    if not distributed:
        #client = Client(processes=False)
        parallel=None
    else:
        cluster = SGECluster(cores=1)
        cluster.start_workers(threads)
        client = Client(cluster) 

        with joblib.parallel_backend('dask', wait_for_workers_timeout=60):
            parallel = joblib.Parallel(verbose=1)

    result = run_stability_experiment(models,datasets,parallel=parallel)

    logger.debug(len(result))
    file,saved_result = save_results([result], filename)
    logger.info("finished job.py with filename {}".format(file))
    logger.info("with end time {}".format(time.ctime()))

    return saved_result,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start experiment manually from CLI")
    parser.add_argument("--tempresfile", type=str)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--toy", type=bool,default=False)
    parser.add_argument("--distributed", type=bool,default=False)
    parser.add_argument("--debug", type=bool,default=False)

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    main_exp(n_bootstraps=args.iters,tempres=args.tempresfile,
        selectmodels=args.models,filename=args.filename,
        threads=args.threads, toy = args.toy, noise=args.noise,
        distributed=args.distributed)

