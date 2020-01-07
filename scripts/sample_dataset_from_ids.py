import argparse
import json, sys, os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', dest='dataset', help='dataset to'
      'sample', required=True, type=str)
  parser.add_argument('-o', '--output', dest='output', help='file to put the'
      'sampled dataset', required=True, type=str)
  parser.add_argument('-i', '--id_list', dest='ids', help='Json list of ids to'
      'sample', required=True, type=str)
  return parser.parse_args()

def sample_from_ids(dataset, ids):
  for article in dataset:
    for p in article['paragraphs']:
      p['qas'] = [q for q in p['qas'] if q['id'] in ids]
  return dataset

def main():
  dataset = json.load(open(FLAGS.dataset, 'r'))['data']
  ids = json.load(open(FLAGS.ids, 'r'))
  dataset = sample_from_ids(dataset, ids)
  json.dump(fp=open(FLAGS.output, 'w'), obj={ 'data': dataset })

if __name__ == '__main__':
  FLAGS = parse_args()
  main()
