from collections import defaultdict
from stat import S_ISFIFO

import math
import os
import json
import sys

def usage():
  print('Usage: ./format_results.py <results> | -- <stdin>')
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

def pretty_print(datapoint, highlith):
  rounded = floor(datapoint)
  if highlith:
    return '\\textbf{' + str(rounded) + '}\t& '
  return '{:<14}\t& '.format(rounded)

def extract_fields_no_empty(data):
  fields = []
  fields.append(data['exact'])
  fields.append(data['NoAns_percentage'])
  return fields

def extract_fields_with_empty(data):
  fields = []
  fields.append(data['exact'])
  fields.append(data['HasAns_exact'])
  fields.append(data['NoAns_exact'])
  fields.append(data['NoAns_percentage'])
  fields.append(data['NoAns_precision'])
  fields.append(data['NoAns_recall'])
  fields.append(data['NoAns_f1'])
  return fields

def extract_fields(row):
  if row.get('NoAns_exact', None) is None:
    fields = extract_fields_no_empty(row)
  else:
    fields = extract_fields_with_empty(row)
  return fields

def process_row(row, highlith_indices):
  pretty_data = []
  for idx, value in enumerate(row):
    pretty_value = pretty_print(value, bool(idx in highlith_indices))
    pretty_data.append(pretty_value)
  print(''.join(pretty_data))

def find_index(column, _max=True):
  sorted_indices = sorted(range(len(column)), reverse=_max, key=column.__getitem__)
  return sorted_indices[:2]

def transpose(data):
  return [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]

def column_to_row_index(column_indices):
  row_indices = defaultdict(list)
  for idx, indices in enumerate(column_indices):
    for index_value in indices:
      row_indices[index_value].append(idx)
  return row_indices

def find_max(data):
  trans_data = transpose(data)
  # get the two max per column
  max_column_indices = [find_index(row) for i, row in
      enumerate(trans_data)]
  return column_to_row_index(max_column_indices)

def main(raw_data):
  data = [extract_fields(row) for row in raw_data]
  row_indices = find_max(data)
  for idx, row in enumerate(data):
    process_row(row, row_indices[idx])

if __name__ == '__main__':
  data = parse_args()
  main(data)
