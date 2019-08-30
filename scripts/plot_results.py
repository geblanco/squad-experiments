from matplotlib import pyplot as plt
from collections import defaultdict

import json, os, sys
import pandas as pd
import argparse

plt.rcParams.update({'font.size': 30})
args = None
main_folder = None
number_seq = ''.join([str(s) for s in list(range(10))])

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--datadir', help='directory with results', required=True, type=str)
  parser.add_argument('-f', '--filter', help='directory to put the clean data', required=False,
                      type=str, action='store', nargs="*", default=[])
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def shorten_name(names):
  names = [n.capitalize() for n in names]
  if len(names) > 1:
    # if more than one name, only initial letter
    names = [n[0] for n in names]
  return names

def sort_index(index):
  ret_index = []
  # Sort like: Newsqa, N + S..., Squad, S + T ...
  # group by initial letter
  groups = defaultdict(list)
  for i in index:
    groups[i[0]].append(i)
  # reorder groups
  for group in groups.values():
    tokens = [t.split(' ') for t in group]
    tokens.sort(reverse=True)
    ret_index.extend([' '.join(t) for t in tokens])
  return ret_index

def clean_name(name): 
  name_tokens = name.split('_') 
  if name_tokens[1] == 'pred' or name_tokens[1] == 'untrained': 
      name_tokens[0:2] = '' 
  elif name_tokens[1] == 'trained' and name_tokens[2] == 'base': 
      name_tokens[1:3] = '' 
  elif name == 'squad_trained_newsqa_triviaqa_model_out': 
      name_tokens = 'squad_newsqa_triviaqa_model_out'.split('_')
  # fixes for names like: mixed_trained_4_epochs_model_out
  if name_tokens[1] == 'trained':
    if name_tokens[2] not in number_seq:
      name_tokens[1] = '+'
    else:
      name_tokens[1:2] = ''
  name_tokens[-2:] = ''
  if name_tokens[-1] == 'epochs':
    name_tokens[-1:] = ''
  # print('Clean %s -> %s' % (name, '_'.join(name_tokens)))
  return ' '.join(shorten_name(name_tokens))

def read_model_data(model_dir):
  data_dirs = [d for d in os.listdir(model_dir) if d.endswith('_out')]
  data_dirs.sort()
  data = { d: json.load(open(model_dir + '/' + d + '/results.json', 'r'))  for d in data_dirs }
  acc_data = { clean_name(d): data[d]['exact'] for d in data.keys() }
  return acc_data

def main():
  data = {}
  dirs = os.listdir(main_folder)
  index = ['newsqa', 'squad', 'triviaqa', 'mixed']
  if len(args.filter) > 0:
    dirs = [d for d in dirs if d in args.filter]
    index = [d for d in dirs if d in args.filter]

  model_data_dirs = ['{}/{}'.format(main_folder, d) for d in dirs]
  for model_data_dir in model_data_dirs:
    model_data = read_model_data(model_data_dir)
    dataset_name = model_data_dir.split('/')[-1]
    data[dataset_name] = model_data

  df = pd.DataFrame(data)
  df = df.reindex(index, axis=1)
  df = df.reindex(sort_index(df.index))
  print(df)
  ax = df.plot(kind='bar')
  ax.set_xticklabels(df.index, rotation='horizontal')
  plt.yticks(list(range(0, 101, 10)))
  plt.grid(axis='y', alpha=0.5)

  # to get a max over each dataset:
  # df.max()
  # to get the rows with the max values
  # df.loc[df.idxmax()]

  plt.show()

if __name__ == '__main__':
  args = parse_args()
  main_folder = args.datadir
  main()