from matplotlib import pyplot as plt
import json, os, sys

if len(sys.argv) < 2:
  print('Usage: plot_results.py <data dir>')
  sys.exit(0)

def clean_exp_name(name):
  parts = name.split('_')
  dataset_name = parts[0]
  trained = parts[1]
  extra = parts[2]
  return trained + '_' + extra

def read_model_data(model_dir):
  data_dirs = [d for d in os.listdir(model_dir) if d.endswith('_out')]
  data_dirs.sort()
  data = { d: json.load(open(model_dir + '/' + d + '/results.json', 'r'))  for d in data_dirs }
  acc_data = { clean_exp_name(d): data[d]['exact'] for d in data.keys() }
  return acc_data

def group_model_data(data_dict):
  # model comes as:
  #  <dataset_name>_(trained/untrained)_<extra_model>_model_out
  dataset_keys = list(data_dict.keys())
  dataset_name = dataset_keys[0].split('_')[0]
  model_types = {'trained': {}, 'untrained': {}}
  for key in dataset_keys:
    model_type = 'untrained'
    if key.find(model_type) == -1:
      model_type = 'trained'
    extra_model = key.replace('{}_{}_'.format(dataset_name, model_type), '').replace('_model_out', '')
    model_types[model_type][extra_model] = data_dict[key]
  return model_types

def plot_model_data(dataset_name, data_dict):
  model_types = group_model_data(data_dict)
  trained_array = list(model_types['trained'].values())
  untrained_array = list(model_types['untrained'].values())

  fig, ax = plt.subplots(figsize=(10,5))
  x = list(range(len(trained_array)))
  width = (100 / (len(x) + 4)) / 100

  positions = [(p + (width * 0)) - width for p in x]
  plt.bar(positions, trained_array, width, alpha=0.75, label='Trained')

  positions = [(p + (width * 1)) - width for p in x]
  plt.bar(positions, untrained_array, width, alpha=0.75, label='Untrained')

  ax.set_ylabel('Exact Matches')
  # plt.ylim([0, 100])
  plt.title(dataset_name)
  plt.yticks(list(range(0, 101, 10)))
  plt.xticks([], [])

  plt.legend(['Trained', 'Untrained'], loc='upper left')
  ax.set_xticks(x)
  ax.set_xticklabels(list(model_types['trained'].keys()))
  plt.grid(axis='y', alpha=0.5)
  plt.show()

data = {}
model_data_dirs = ['{}/{}'.format(sys.argv[1], d) for d in os.listdir(sys.argv[1])]
for model_data_dir in model_data_dirs:
  model_data = read_model_data(model_data_dir)
  dataset_name = model_data_dir.split('/')[-1]
  plot_model_data(dataset_name, model_data)
