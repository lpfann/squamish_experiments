# Squamish Experiments
This repository contains experiments for the scientific publication about [Squamish](https://github.com/lpfann/squamish).
Squamish is a new feature selection tool to find all-relevant features and their relevance classes.

```bibtex
@misc{pfannschmidt2020sequential,
    title={Sequential Feature Classification in the Context of Redundancies},
    author={Lukas Pfannschmidt and Barbara Hammer},
    year={2020},
    eprint={2004.00658},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Preprints can be found at https://pub.uni-bielefeld.de/record/2942271 or https://arxiv.org/abs/2004.00658.


# Replicate Experiments
To replicate the experimental results of the paper (figure and tables) we provide a docker image and several scripts to produce a (hopefully) identical output.

Build the image with
```sh
docker build -t squamish_experiments .
```
and then run
```sh
docker run -v ./tmp:/exp/tmp:Z -v ./output:/exp/output:Z -it squamish_experiments make 
```
which calls make inside the container to execute all experiments in the `Makefile`.

After the experiments are done (can take several hours) the output should end up in the `./output` folder.

It's possible to change the following parameters as environment variables in the docker command via the `-e` option:

Defaults used in paper
- SEED = 123
- REPEATS = 10
- N_THREADS = 1