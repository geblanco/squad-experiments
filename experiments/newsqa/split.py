from collections import defaultdict
from os.path import join

import argparse
import json
import sys

def write_json(version, data, path):
  json.dump(fp=open(path, 'w'), obj={ version: version, data: data }, \
    ensure_ascii=False, indent=2)

def split(dataset_path, output_dir):
  data = json.load(open(dataset_path, 'r'))
  dataset = defaultdict(list)
  # datapoint['type'] is train, dev or test
  for datapoint in data['data']:
    dataset[datapoint['type']].append(datapoint)

  write_json(data['version'], join(output_dir, 'train.json'), dataset['train'])
  write_json(data['version'], join(output_dir, 'dev.json'), dataset['dev'])
  write_json(data['version'], join(output_dir, 'test.json'), dataset['test'])

if __name__ == '__main__':
  dir_name = os.path.dirname(os.path.abspath(__file__))
  default_output_dir = join(dir_name, 'split_data')

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', default=join(dir_name, 'combined-newsqa-data-v1.json'),
                      help="The path to the News QA dataset (combined-newsqa-data-v1.json).")
  parser.add_argument('--output_dir_path', '--output_dir', default=default_output_dir,
                      help="The path folder to put the split up data. Default: %s" % default_output_dir)
  args = parser.parse_args()

  split(args.dataset_path, args.output_dir_path)