#!/usr/bin/env python

"""
Cross answers from different models, this gets an amount of overlap between
different models against the same dataset.
"""
from collections import defaultdict
import argparse, json, os, sys

args = None

def parse_args():
  parser = argparse.ArgumentParser('Cross answers between models over the same dataset.')
  parser.add_argument('--answers', '-a', help='Input answers JSON file. ' +
                      'A dictionary with question ids and correct or not flag. ' +
                      'Can be obtained with the modified evaluate SQuAD script.',
                      action='store', default=[], required=True, type=str, nargs='*')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  overlap = { k: { w: None for w in args.answers } for k in args.answers}
  answers = [json.load(open(a, 'r')) for a in args.answers]
  # check number of answers matches
  assert(len(set([len(a) for a in answers])) == 1)
  for answer_set in zip(*answers):

if __name__ == '__main__':
  args = parse_args()
  main()