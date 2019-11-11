from collections import Counter

import argparse
import random
import json
import sys
import os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--datasets', dest='datasets', action='store',
      help='datasets to compare topics', required=True, type=str, nargs='*')
  return parser.parse_args()

def get_topics(dataset):
  topics = [article['title'] for article in dataset]
  return topics

def main(files):
  topics = []
  for dataset_file in files:
    dataset = json.load(open(dataset_file, 'r'))['data']
    topics.extend(get_topics(dataset))
  count = Counter(topics)
  rep_count = [item for item in count if count[item] > 1]
  print(f'Repetead topics: {rep_count}')

if __name__ == '__main__':
  FLAGS = parse_args()
  main(FLAGS.datasets)

