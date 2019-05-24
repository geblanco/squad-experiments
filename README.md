# SQuAD-experiments
Repository to replicate [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) experiments and results

# Setup
`prepare.sh` script setups bert with it's model and SQuAD. It requires `git`, `wget`, `unzip` and `pip` commands. Python packages are setup through pipenv (which is also installed).

Experiments are setup through \*.experiment files (see example.experiment for further information), they state the location of the necessary resources and some model settings.

# Run
To run an experiment, copy the experiment file to the root of this project (in the future could be done automatically) and run the experimentation script inside the virtualenv:

```bash
$(python -m site --user-base)/bin/pipenv run ./run.sh
```

# Notes
Some datasets may require some manual setup in order to work. For example [News QA](https://github.com/Maluuba/newsqa) requires manual download (due to license issues) and docker for compilation.
