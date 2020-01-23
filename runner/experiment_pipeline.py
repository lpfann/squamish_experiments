import time
from collections import defaultdict
import dill as pickle
import joblib
from numpy.random import RandomState
from dask_jobqueue import SGECluster
from dask.distributed import Client
import dask
from job import Job

# import fsmodel
import import_data
import argparse
from fsmodel import RF, SQ, FRI, LM
import logging

logger = logging.getLogger("Experiment")

## functions which are run in the worker threads for parallel computation
def worker_stability(job: Job):
    return job.run_one_bootstrap()


def worker_performance(job: Job):
    return job.run_one_perf_test()


## Main task delegation process  - only in main
def run_stability_experiment(models, datasets, parallel=None):
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
        results = list(map(worker_stability, sets))
    else:
        results = parallel(map(joblib.delayed(worker_stability), sets))

    if len(results) < 1:
        raise Exception()

    return results


def run_performance_experiment(stabResults, parallel, datasets):
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


def get_models(seed):
    # FRI
    fri_model_exc = FRI(random_state=seed)
    # ElasticNet
    eelm = LM(random_state=seed)
    sq = SQ(random_state=seed)
    rf = RF(random_state=seed)

    models = {
         "FRI": fri_model_exc,
        "ElasticNet": eelm,
         "RF": rf,
         "SQ": sq,
        # "AllFeatures": afm,
    }
    return models


def get_datasets(seed):
    logging.debug("FIBROSIS")
    set_fibrosis = import_data.import_Fibrosis(seed)
    logging.debug("FLIP")
    set_flip = import_data.import_FLIP(seed)
    logging.debug("T21")
    set_t21 = import_data.import_T21(seed)
    logging.debug("SPECTF")
    set_spectf = import_data.import_SPECTF(seed)
    logging.debug("WBC")
    set_WBC = import_data.import_wbc(seed)
    logging.debug("colposcopy")
    set_col = import_data.import_colposcopy(seed)
    logging.debug("cervical")
    set_cervical = import_data.import_cervical(seed)

    datasets = {
        # "fibrosis": set_fibrosis,
        # "colposcopy": set_col,
        "cervical": set_cervical,
        "flip": set_flip,
        "t21": set_t21,
        "spectf": set_spectf,
        "wbc": set_WBC,
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
    "Set8": {"n": 2000, "strong": 10, "weak": 10, "irr": 50},
    #"Set9": {"n": 5000, "strong": 10, "weak": 20, "irr": 200},
}


def get_toy_datasets(seed, toy_set_params=toy_set_params, noise=0.0):
    datasets = {}
    for name, params in toy_set_params.items():
        n_strel = params["strong"]
        n_redundant = params["weak"]
        n_irrel = params["irr"]
        n_features = n_strel + n_redundant + n_irrel
        n_samples = params["n"]
        datasets[name] = import_data.Toyset(
            n_features=n_features,
            n_strel=n_strel,
            n_redundant=n_redundant,
            n_samples=n_samples,
            random_state=seed,
            noise=noise,
        )
    return datasets


def main_exp(
    n_bootstraps=25,
    SEED=RandomState(1337),
    tempres=None,
    selectmodels=None,
    filename="test",
    threads=8,
    toy=True,
    noise=0,
    distributed=False,
):
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
        datasets = get_toy_datasets(SEED, noise=noise)
    else:
        datasets = get_datasets(SEED)
    # datasets.update(datasets_toy)

    # if toy:
    #    datasets = get_bootstrapped_datasets(datasets, n_bootstraps)

    datasets = get_bootstrapped_datasets(datasets, n_bootstraps)

    # Get models used in testing
    models = get_models(SEED)
    if selectmodels is not None:
        models = {k: v for k, v in models.items() if k in selectmodels}
    logger.info(f"models:{models}")

    # Cluster
    if not distributed:
        # client = Client(processes=False)
        parallel = None
    else:
        cluster = SGECluster(cores=1)
        cluster.start_workers(threads)
        client = Client(cluster)

        with joblib.parallel_backend("dask", wait_for_workers_timeout=60):
            parallel = joblib.Parallel(verbose=1)

    result = run_stability_experiment(models, datasets, parallel=parallel)

    logger.debug(len(result))
    file, saved_result = save_results([result], filename)
    logger.info("finished job.py with filename {}".format(file))
    logger.info("with end time {}".format(time.ctime()))

    return saved_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start experiment manually from CLI")
    parser.add_argument("--tempresfile", type=str)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--toy", type=bool, default=False)
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    main_exp(
        n_bootstraps=args.iters,
        tempres=args.tempresfile,
        selectmodels=args.models,
        filename=args.filename,
        threads=args.threads,
        toy=args.toy,
        noise=args.noise,
        distributed=args.distributed,
    )

