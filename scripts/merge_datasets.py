import random, argparse
import json, sys

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', action='store', dest='datasets', help='list of datasets to merge', required=True, type=str, nargs="*")
parser.add_argument('-o', '--output', help='directory to put the merged data', required=True, type=str)
args = parser.parse_args()

random.seed(42)

output_dataset = []
# import data
for data_dir in args.datasets:
  dataset = json.load(open(data_dir, 'r'))
  output_dataset.extend(dataset['data'])

# shuffle
random.shuffle(output_dataset)
json.dump(fp=open(args.output, 'w'), obj={ 'version': 'v2.1', 'data': output_dataset })
