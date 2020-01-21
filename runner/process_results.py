import argparse
import pathlib
import dill as pickle
import logging

def remove_bad_jobs(jobs):
    lenght = len(jobs)
    removed_zerosets = list(filter(lambda j: sum(j.featset)>0,jobs))
    n_removed = lenght- len(removed_zerosets)
    print("Removed {} invalid jobs".format(n_removed))
    return removed_zerosets

def uniq_models(r):
    models = [j.modelname for j in r]
    models = set(models)
    return models

def uniq_setname(r):
    setname = [j.setname for j in r]
    setname = set(setname)
    return setname

def print_stats(r):
    print("Length:", len(r))
    models = uniq_models(r)
    sets = uniq_setname(r)
    print("Models:",models)
    print("Sets:",sets)
    for m in models:
        m_list = filter_model(r,m)
        print(len(m_list))

def print_job(j):
    selected = sum(j.featset)
    print("{} - {} - n_selected:{}".format(j.modelname, j.setname, selected))

def check_selected_n(result):
    for j in result:
        if sum(j.featset)<1:
            print_job(j)
def filter_model(r, name):
    return list(filter(lambda j: j.modelname == name, r))


def main(path):


    filepath = pathlib.Path(path)
    if not filepath.exists():
        logging.error("File does not exist.")
        raise Exception()

    with filepath.open('rb') as f:
        result = pickle.load(f)


    print_stats(result)
    res_removed = remove_bad_jobs(result)
    print(len(res_removed))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process results')
    parser.add_argument("--file", type=str,default="./results/tempres_1533421658.1877384.dat")
    args = parser.parse_args()

    main(args.file)