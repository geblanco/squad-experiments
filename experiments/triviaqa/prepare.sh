#!/bin/bash

# whether to only prepare the file system or
# do the full process
only_fs=${1:-0}

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

# create destiny dataset dir if necessary 
exists $DST_DATASET || mkdir $DST_DATASET

if [[ $only_fs -eq 1 ]]; then
  exit 0
fi

cd $BASEDIR

# get conversion script
exists $REPO_DIR || git clone --branch=convert-to-squad-2 https://github.com/Chubasik/triviaqa.git

# donwload dataset
if [[ ! -d $SRC_DATASET ]]; then
  echo "Downloading dataset..."
  wget -q http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
  echo "Uncompressing dataset..."
  exists || mkdir $SRC_DATASET
  tar xfz triviaqa-rc.tar.gz -C $SRC_DATASET
  echo "done"
fi

cd $REPO_DIR

pipenv install --python 3 --skip-lock -r requirements.txt

pipenv run python3 -m nltk.downloader punkt
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/wikipedia-dev.json    --squad_file $DST_DATASET/wikipedia-dev.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/web-dev.json          --squad_file $DST_DATASET/web-dev.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/verified-web-dev.json --squad_file $DST_DATASET/verified-web-dev.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/wikipedia-train.json  --squad_file $DST_DATASET/wikipedia-train.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/web-train.json        --squad_file $DST_DATASET/web-train.json
pipenv run python3 convert_to_squad2_format.py --wikipedia_dir $SRC_DATASET/evidence/wikipedia --web_dir $SRC_DATASET/evidence/web --triviaqa_file $SRC_DATASET/qa/verified-wikipedia-dev.json --squad_file $DST_DATASET/verified-wikipedia-dev.json

cd $cwd
