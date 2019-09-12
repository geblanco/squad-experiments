"""
Get some metrics about the incremental experiments:
  Average EM
  Average F1
  Results plotted accross increments, one per dataset
"""
from typing import NamedTuple
from matplotlib import pyplot as plt
import json, argparse, os
import re

args = None

class Metric(NamedTuple):
  has_ans_em: list = []
  has_ans_f1: list = []
  empty_em: list = []
  empty_f1: list = []

def mean(lst):
  return sum(lst)/len(lst)

def natural_sort_key(s):
  nsre = re.compile('([0-9]+)')
  return [int(text) if text.isdigit() else text.lower()
      for text in _nsre.split(s)]

def parse_args():
  parser = argparse.ArgumentParser('Analyse results from incremental experiments.')
  parser.add_argument('--filelist', '-f', help='Filelist with directories of \
      incremental experiments', default=None, required=True, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def process_result_file(file, store):
  result = json.load(open(file, 'r'))
  store.has_ans_em.append(result['HasAns_exact'])
  store.has_ans_f1.append(result['HasAns_f1'])
  store.empty_em.append(result['NoAns_exact'])
  store.empty_f1.append(result['NoAns_f1'])

def average_results(filelist):
  ret = Metric()

  # there must exist a resutls file
  for line in open(filelist, 'r').read():
    file_metric = Metric()
    files = os.listdir(line)
    files.sort(key=natural_sort_key)
    files = [f for f in files if f.startswith('result')]
    for f in files:
      process_result_file(f, file_metric)
    ret.has_ans_em.append(mean(file_metric.has_ans_em))
    ret.has_ans_f1.append(mean(file_metric.has_ans_f1))
    ret.empty_em.append(mean(file_metric.empty_em))
    ret.empty_f1.append(mean(file_metric.empty_f1))
  return ret

def main():
  results = average_results(args.filelist)

if __name__ == '__main__':
  args = parse_args()
  main()
