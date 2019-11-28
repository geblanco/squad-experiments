from stat import S_ISFIFO

import math
import os
import json
import sys

def usage():
  print('Usage: ./format_results.py <dataset> | -- <stdin>')
  sys.exit(0)

def parse_args():
  if len(sys.argv) < 2:
    usage()

  data = None
  # allow error to be raised
  argv_abspath = os.path.abspath(sys.argv[1])
  if os.path.exists(argv_abspath):
    # input is a file
    data = json.load(open(argv_abspath, 'r'))
  elif S_ISFIFO(os.fstat(0).st_mode):
    # piped process, consume stdin
    data = json.loads(''.join(sys.stdin.readlines()))
  else:
    usage()
  return data

def floor(number):
  return math.floor(number * 100)/100.0

def pretty_print(datapoint):
  rounded = floor(datapoint)
  return '{}\t'.format(rounded)

def format_no_empty(data):
  pretty_data = []
  pretty_data.append(pretty_print(data['exact']))
  pretty_data.append(pretty_print(data['NoAns_percentage']))
  return pretty_data

def format_with_empty(data):
  pretty_data = []
  pretty_data.append(pretty_print(data['exact']))
  pretty_data.append(pretty_print(data['HasAns_exact']))
  pretty_data.append(pretty_print(data['NoAns_exact']))
  pretty_data.append(pretty_print(data['NoAns_percentage']))
  pretty_data.append(pretty_print(data['NoAns_precision']))
  pretty_data.append(pretty_print(data['NoAns_recall']))
  pretty_data.append(pretty_print(data['NoAns_f1']))
  return pretty_data

def main(data):
  if data.get('NoAns_exact', None) is None:
    pretty_data = format_no_empty(data)
  else:
    pretty_data = format_with_empty(data)
  print(''.join(pretty_data))

if __name__ == '__main__':
  data = parse_args()
  main(data)
