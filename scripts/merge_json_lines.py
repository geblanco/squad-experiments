import argparse
import json, sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True, type=str,
      help='Where to store the results.')

  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  data_lines = [json.loads(''.join(line)) for line in sys.stdin.readlines()]
  json.dump(obj=data_lines, fp=open(OPTS.output, 'w'))

if __name__ == '__main__':
  OPTS = parse_args()
  main()
