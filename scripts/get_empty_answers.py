#!/usr/bin/env python

"""
Get all ids of answers without response.
"""
import json, os, sys
import argparse

args = None
input_data = []

def parse_args():
  parser = argparse.ArgumentParser('Get all ids of answers without response.')
  parser.add_argument('--data', '-d', help='Input json data',
                      default=None, required=True, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  empty_ids = {}
  for entry in input_data:
    for paragraph in entry['paragraphs']:
      for qa in paragraph['qas']:
        qas_id = qa['id']
        question_text = qa['question']
        empty = 0
        if qa.get('is_impossible', False) or len(qa['answers']) == 0:
          empty = 1
        empty_ids[qas_id] = empty
  print(empty_ids)

if __name__ == '__main__':
  args = parse_args()
  input_data = json.load(open(args.data, 'r'))['data']
  main()
