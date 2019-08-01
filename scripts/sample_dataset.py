import random, argparse
import json, sys, math, os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', dest='dataset', help='dataset to sample', required=True, type=str)
parser.add_argument('-o', '--output', dest='output', help='file to put the sampled dataset', required=True, type=str)
parser.add_argument('-s', '--sample', dest='sample', help='sample size in percentage', required=True, type=int)
args = parser.parse_args()

sample_size = args.sample % 100

if sample_size == 0:
  sample_size = 100

dataset = json.load(open(args.dataset, 'r'))['data']
dataset_len = len(dataset)

total_size = math.ceil((dataset_len * sample_size) / 100)
dataset = random.sample(dataset, total_size)

json.dump(fp=open(args.output, 'w'), obj={ 'data': dataset })
