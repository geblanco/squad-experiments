from matplotlib import pyplot as plt
import json, os, sys
import pandas as pd

# if len(sys.argv) < 2:
#   print('Usage: plot_results.py <data dir>')
#   sys.exit(0)
# 
# main_folder = sys.argv[1]
main_folder = 'results'
number_seq = ''.join([str(s) for s in list(range(10))])

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
  return '_'.join(name_tokens)

def read_model_data(model_dir):
  data_dirs = [d for d in os.listdir(model_dir) if d.endswith('_out')]
  data_dirs.sort()
  data = { d: json.load(open(model_dir + '/' + d + '/results.json', 'r'))  for d in data_dirs }
  acc_data = { clean_name(d): data[d]['exact'] for d in data.keys() }
  return acc_data

data = {}
model_data_dirs = ['{}/{}'.format(main_folder, d) for d in os.listdir(main_folder)]
for model_data_dir in model_data_dirs:
  model_data = read_model_data(model_data_dir)
  dataset_name = model_data_dir.split('/')[-1]
  data[dataset_name] = model_data

df = pd.DataFrame(data)
df = df.reindex(['newsqa', 'squad', 'triviaqa', 'mixed'], axis=1)
print(df)
ax = df.plot(kind='bar')
ax.set_xticklabels(df.index, rotation=20)
plt.yticks(list(range(0, 101, 10)))
plt.grid(axis='y', alpha=0.5)

plt.show()