#!/bin/bash

# whether to only prepare the file system or
# do the full process
only_fs=${1:-0}

cwd=`pwd`

EXPERIMENTS_DIR=`dirname $PWD`
ROOT_DIR=`dirname $EXPERIMENTS_DIR`
SRC_TRAIN_DATASETS="${EXPERIMENTS_DIR}/newsqa/data/newsqa_squad/train.json ${EXPERIMENTS_DIR}/triviaqa/data/triviaqa_squad/wikipedia-train.json ${ROOT_DIR}/squad/train-v2.0.json"
SRC_DEV_DATASETS="${EXPERIMENTS_DIR}/newsqa/data/newsqa_squad/dev.json ${EXPERIMENTS_DIR}/triviaqa/data/triviaqa_squad/wikipedia-dev.json ${ROOT_DIR}/squad/dev-v2.0.json"
SRC_TEST_DATASETS="${EXPERIMENTS_DIR}/newsqa/data/newsqa_squad/test.json ${EXPERIMENTS_DIR}/triviaqa/data/triviaqa_squad/wikipedia-test.json ${ROOT_DIR}/squad/test-v2.0.json"

BASEDIR=`pwd`/data
DST_DATASETS_DIR=${BASEDIR}/mixed_squad
DST_TRAIN_DATASET=${BASEDIR}/mixed_squad/train.json
DST_DEV_DATASET=${BASEDIR}/mixed_squad/dev.json
DST_TEST_DATASET=${BASEDIR}/mixed_squad/test.json

echo "Directories"
echo "BASEDIR=$BASEDIR"
echo "SRC_TRAIN_DATASETS=$SRC_TRAIN_DATASETS"
echo "SRC_DEV_DATASETS=$SRC_DEV_DATASETS"
echo "SRC_TEST_DATASETS=$SRC_TEST_DATASETS"
echo "DST_DATASETS_DIR=$DST_DATASETS_DIR"
echo "DST_TRAIN_DATASET=$DST_TRAIN_DATASET"
echo "DST_DEV_DATASET=$DST_DEV_DATASET"
echo "DST_TEST_DATASET=$DST_TEST_DATASET"

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
exists $DST_DATASETS_DIR || mkdir $DST_DATASETS_DIR

if [[ $only_fs -eq 1 ]]; then
  exit 0
fi

echo "Merging $SRC_TRAIN_DATASETS"
python ../merge_datasets.py --datasets $SRC_TRAIN_DATASETS --output $DST_TRAIN_DATASET
echo "Merging $SRC_DEV_DATASETS"
python ../merge_datasets.py --datasets $SRC_DEV_DATASETS --output $DST_DEV_DATASET
echo "Merging $SRC_TEST_DATASETS"
python ../merge_datasets.py --datasets $SRC_TEST_DATASETS --output $DST_TEST_DATASET
