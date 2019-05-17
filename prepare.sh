#!/bin/bash

# exit on error
set -e

# download bert repo
git clone https://github.com/google-research/bert.git
cd bert
patch -p1 < ../run_squad.patch
cd -

mkdir squad
# download squad files
wget -q -O squad/train-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -q -O squad/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
wget -q -O squad/evaluate-v2.0.py https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

mkdir models
cd models
model='uncased_L-12_H-768_A-12.zip'
# download bert models, light, uncased english model
wget -q https://storage.googleapis.com/bert_models/2018_10_18/$model
unzip $model
mv "${model%.*}" uncased
cd -

# bert output
mkdir squad_output

# install pipenv
pip install --user pipenv
export PATH=$(python -m site --user-base)/bin:$PATH

pipenv install --skip-lock --python 3.6 -r bert/requirements.txt

