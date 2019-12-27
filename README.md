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


# Scripts

Info about some of the scripts:
- `scripts/eval.py`: Calculates precision/recall/f1 over empty answers
- `scripts/format_results.py`: Format the results for latex tables
- `scripts/split_dataset.py`: Splits a SQuAD like dataset into desired sizes.

Split train dataset into three splits, representing 60, 20 and 20 percent of the original one.

`python scripts/split_dataset.py -d squad/train-v2.0.json -s -o squad/ -p 60 20 20 --names 'train-v2.1' 'dev-v2.1' 'test-v2.1'`

Example to get latex-table ready data

```bash
# get results, evaluated with squad script
./scripts/eval_experiments.sh filelist

# get results for the last model in one json file (triviaqa model)
for f in $(head -n 126 filelist | tail -n 12 | grep -v "#--"); do 
expfile=$(echo $f | sed -e 's/\.\/results/experiments/' | sed -e 's/_out/\.experiment/')
dataset=$(grep -Po '(?<=PREDICT_FILE=`pwd`/).*$' $expfile)
python scripts/eval.py $dataset $f/predictions.json --merge $f/results.json
done | python scripts/merge_json_lines.py --output /tmp/triviaqa_results.json

# pretty print for latex
python ./scripts/format_results.py /tmp/triviaqa_results.json
```
