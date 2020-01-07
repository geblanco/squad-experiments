import argparse
import json, sys, os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--predictions', dest='preds', help='predictions to'
      'extract ids from', required=True, type=str)
  parser.add_argument('-o', '--output', dest='output', help='file to put the'
      'list of ids', required=True, type=str)
  return parser.parse_args()

def main():
  dataset = json.load(open(FLAGS.preds, 'r'))
  ids = list(dataset.keys())
  json.dump(fp=open(FLAGS.output, 'w'), obj=ids)

if __name__ == '__main__':
  FLAGS = parse_args()
  main()
