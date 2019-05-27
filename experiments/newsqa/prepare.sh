#!/bin/bash

cwd=`pwd`
DATASET_CONVERTER_PATH=$(realpath ../..)/converter

BASEDIR=`pwd`/data
SRC_DATASET="${BASEDIR}/newsqa/maluuba/newsqa"
DST_DATASET=${BASEDIR}/newsqa_squad
REPO_DIR="${BASEDIR}/newsqa"
EXTRACTED_DATASET="${REPO_DIR}/combined-newsqa-data-v1.json"

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

cd $BASEDIR

# get conversion script
exists $REPO_DIR || git clone https://github.com/Maluuba/newsqa

# donwload dataset
if [[ (! -f $SRC_DATASET/newsqa-data-v1.tar.gz) || (! -f $SRC_DATASET/cnn_stories.tgz) ]]; then
  echo "Download newsqa and cnn stories datasets from: https://github.com/Maluuba/newsqa and come back"
  exit 0
fi

set -e

cd $REPO_DIR

if [[ ! -f $REPO_DIR/combined-newsqa-data-v1.json ]]; then
  # Build newsqa dataset (Taken from README)
  echo "Building docker image..."
  sudo docker build -t maluuba/newsqa .
  echo "Compiling dataset..."
  sudo docker run --rm -it -v ${PWD}:/usr/src/newsqa --name newsqa maluuba/newsqa

  # You now have the datasets. See
  # combined-newsqa-data-*.json,
  # combined-newsqa-data-*.csv,
fi

if [[ ! -f $REPO_DIR/split_data/train.csv ]]; then
  # Tokenize, optional
  echo "Splitting dataset..."
  sudo docker run --rm -it -v ${PWD}:/usr/src/newsqa --name newsqa maluuba/newsqa /bin/bash --login -c 'python maluuba/newsqa/data_generator.py'
  # maluuba/newsqa/newsqa-data-tokenized-*.csv.

  echo "Changing ownership over data after docker operations..."
  sudo chown -R $USER $REPO_DIR
fi

_log_path=${BASEDIR}/conversion.log
_data_path=${REPO_DIR}/split_data
# extract data to make it available for the conversor
# tar xfz $SRC_DATASET/cnn_stories.tgz -C $_data_path

cd $DATASET_CONVERTER_PATH
# convert to squad
# train
pipenv run python executor.py \
  --log_path=$_log_path  \
  --data_path=$_data_path \
  --from_files="source:train.csv, story:cnn/stories/" \
  --from_format="newsqa"  \
  --to_format="squad" \
  --to_file_name="train.json"

# dev
pipenv run python executor.py \
  --log_path=$_log_path  \
  --data_path=$_data_path \
  --from_files="source:dev.csv, story:cnn/stories/" \
  --from_format="newsqa"  \
  --to_format="squad" \
  --to_file_name="dev.json"

# test
pipenv run python executor.py \
  --log_path=$_log_path  \
  --data_path=$_data_path \
  --from_files="source:test.csv, story:cnn/stories/" \
  --from_format="newsqa"  \
  --to_format="squad" \
  --to_file_name="test.json"

mv $_data_path/newsqa_to_squad_train.json $DST_DATASET/train.json
mv $_data_path/newsqa_to_squad_dev.json $DST_DATASET/dev.json
mv $_data_path/newsqa_to_squad_test.json $DST_DATASET/test.json

cd $cwd
