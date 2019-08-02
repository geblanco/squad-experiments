# SQuAD-experiments
Repository to replicate [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) experiments and results

# Setup
`prepare.sh` script setups bert with it's model and SQuAD. It requires `git`, `wget`, `unzip` and `pip` commands. Python packages are setup through pipenv (which is also installed).

Experiments are setup through \*.experiment files (see example.experiment for further information), they state the location of the necessary resources and some model settings.

# Run
To run an experiment launch the run script with the associated experiment file. You can run it locally (inside the virtualenv) with:


```bash
$(python -m site --user-base)/bin/pipenv run ./run.sh <experiment>.experiment
```

Or run it inside a nvidia-docker container (the experiment file should expose `DOCKERIZE=1`). 

```bash
./run.sh <experiment>.experiment
```

To get more details about the available options please see the example.experiment.

# Notes
Some datasets may require some manual setup in order to work. For example [News QA](https://github.com/Maluuba/newsqa) requires manual download (due to license issues) and docker for compilation.

The original BERT repository does not parallelize over GPU, to do so, you can enable [horovod](https://github.com/horovod/horovod) by setting `HOROVOD=1`, defaults to 2 GPU and can be customized inside the `run_squad.sh` script.
