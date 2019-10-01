import random, argparse
import json, sys, math, os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', metavar='dataset', help='dataset to split', required=True, type=str)
parser.add_argument('-o', '--output-dir', metavar='output-dir', help='directory to put the splitted data', required=True, type=str)
parser.add_argument('-s', '--shuffle', action='store_true',
  help='whether to shuffle before split (default: false)to put the splitted data')
parser.add_argument('-p', '--proportions', action='store', nargs="*",
  help='proportions to split the dataset (should sum up to a multple of 100)', required=True, type=int)
parser.add_argument('-n', '--names', action='store', nargs='*', required=False, type=str,
  help='name for each split')
args = parser.parse_args()

total = sum(args.proportions)

if (total % 100) != 0:
  raise ValueError('Bad proportions given')

dataset = json.load(open(args.dataset, 'r'))['data']
dataset_len = len(dataset)

if args.shuffle:
  random.seed(42)
  random.shuffle(dataset)

curr_idx = 0
for idx, split in enumerate(args.proportions):
  norm_split = (split * 100) / total
  amount = math.floor(dataset_len * (norm_split / 100)) + curr_idx
  if idx == (len(args.proportions) -1):
    amount = len(dataset)
  split_name = ('split_{}'.format(idx, amount) if args.names is None else args.names[idx]) + '.json'
  path = os.path.join(args.output_dir, split_name)
  print('Dropping split with proportion {} in {}'.format(split, path))
  json.dump(fp=open(path, 'w'), obj={ 'version': 'v2.1', 'data': dataset[curr_idx:amount] })
  curr_idx = amount
