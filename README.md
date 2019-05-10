# squad-experiments
Repository to replicate [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) experiments and results

# Setup
`prepare.sh` script setups bert with it's model and squad. It requires git, wget, unzip and pip commands. Python packages are setup through pipenv (which is also installed).

# Run
Once prepared, enter the virtual env and start training.

```bash
$(python -m site --user-base)/bin/pipenv shell
./run.sh
```
