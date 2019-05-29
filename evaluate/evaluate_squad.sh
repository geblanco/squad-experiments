#!/bin/bash

python squad/evaluate-v2.0.py experiments/triviaqa_experiment/data/triviaqa_squad/wikipedia-dev.json experiments/triviaqa_experiment/triviaqa_trained_out/predictions.json
