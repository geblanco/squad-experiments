import math
import json
import argparse

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--infiles', '-i', action='store', nargs='*', 
      required=True, help='Input file(s) to extract fields from')
  parser.add_argument('--fields', '-f', action='store', nargs='*',
      required=True, help='Fields to extract')
  parser.add_argument('--latex', '-l', action='store_true', required=False,
      default=False, help='Whether to output data as a latex table or just tab'
      'separated (default)')
  parser.add_argument('--header', '-ph', action='store_true', required=False,
      default=False, help='Print header fields')
  return parser.parse_args()

def floor(number):
  return math.floor(number * 100)/100.0

def format_row(row):
  floored_row = [floor(num) if isinstance(num, float) else str(num) for num in row]
  sep = '&' if FLAGS.latex else ''
  fmt = '{:<14}' if FLAGS.latex else '{}'
  return f'\t{sep}'.join([fmt.format(num) for num in floored_row])

def extract_fields(data, fields):
  outdata = [data.get(field, '-') for field in fields]
  return outdata

def main():
  infiles = [json.load(open(f, 'r')) for f in FLAGS.infiles]
  rows = [format_row(extract_fields(data, FLAGS.fields)) for data in infiles]
  if FLAGS.header:
    print('\t'.join(FLAGS.fields))
  print('\n'.join(rows))

if __name__ == '__main__':
  FLAGS = parse_args()
  main()
