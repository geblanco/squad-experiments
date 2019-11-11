import argparse
import random
import json
import sys
import os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', help='dataset to sample', 
    required=True, type=str)
  parser.add_argument('-o', '--output', dest='output', required=True, type=str,
    help='file to put the sampled dataset')
  parser.add_argument('-p', '--proportions', action='store', nargs="*",
    help='proportions of empty answers to split the dataset (should sum up to a multple of 100)', 
    required=True, type=int)
  return parser.parse_args()

def topics_to_num_noans(dataset):
  no_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        no_ans[article['title']] += int(not bool(qa['answers']))
  return no_ans

def split_dataset(dataset, proportions):
  # gather empty answers
  no_ans_count = topics_to_num_noans(dataset)

def save_dataset(path, dataset):
  with open(path, 'w') as f:
    json.dump(fp=f, obj={ 'data': dataset })

def main(dataset_file, proportions):
  dataset = json.load(open(dataset_file, 'r'))['data']
  dataset_len = len(dataset)
  train, dev = split_dataset(dataset, proportions)
  train_name = os.path.join(output, f'squad-train-{proportions[0]}.json'),
  dev_name = os.path.join(output, f'squad-dev_-{proportions[0]}.json'),
  save_dataset(train_name, train)
  save_dataset(dev_name, dev)

if __name__ == '__main__':
  FLAGS = parse_args()
  total = sum(FLAGS.proportions)
  if (total % 100) != 0:
    raise ValueError('Bad proportions given')
  main(FLAGS.dataset, FLAGS.proportions)

