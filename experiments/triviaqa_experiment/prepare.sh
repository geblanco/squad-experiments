#!/bin/bash

cwd=`pwd`
BASEDIR=`pwd`/data
SRC_DATASET="${BASEDIR}/triviaqa-rc"
DST_DATASET=${BASEDIR}/triviaqa_squad
REPO_DIR="${BASEDIR}/triviaqa"

echo "Directories"
echo "BASEDIR=$BASEDIR"
echo "SRC_DATASET=$SRC_DATASET"
echo "DST_DATASET=$DST_DATASET"
echo "REPO_DIR=$REPO_DIR"

exists() {
  if [[ -d $1 ]]; then
    true
  else
    false
  fi
}

# create an output directory for each experiment
experiment_files=$(ls *.experiment)
for file in ${experiment_files[@]}; do
  output="${file%.*}_out"
  exists $output || mkdir $output
done

# create data dir if necessary
exists $BASEDIR || mkdir $BASEDIR

# get conversion script
exists $REPO_DIR || git clone --branch=convert-to-squad-2 https://github.com/Chubasik/triviaqa.git

# create destiny dataset dir if necessary 
exists $DST_DATASET || mkdir $DST_DATASET

cd $BASEDIR

# donwload dataset
if [[ ! -d $SRC_DATASET ]]; then
  echo "Downloading dataset..."
  wget -q http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
  echo "Uncompressing dataset..."
  tar xfz triviaqa-rc.tar.gz
  echo "done"
fi

cd $REPO_DIR

pipenv install --python 3.6 --skip-lock -r requirements.txt

pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/wikipedia-dev.json    --squad_file $DST_DATASET/wikipedia-dev.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/web-dev.json          --squad_file $DST_DATASET/web-dev.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/verified-web-dev.json --squad_file $DST_DATASET/verified-web-dev.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/wikipedia-train.json  --squad_file $DST_DATASET/wikipedia-train.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/web-train.json        --squad_file $DST_DATASET/web-train.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/verified-wikipedia-dev.json --squad_file $DST_DATASET/verified-wikipedia-dev.json

cd $cwd
