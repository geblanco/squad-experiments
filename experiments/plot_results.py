from matplotlib import pyplot as plt
import json, os, sys
import pandas as pd

if len(sys.argv) < 2:
  print('Usage: plot_results.py <data dir>')
  sys.exit(0)

def clean_name(name): 
  name_tokens=name.split('_') 
  if name_tokens[1] == 'pred' or name_tokens[1] == 'untrained': 
      name_tokens[0:2] = '' 
  elif name_tokens[1] == 'trained' and name_tokens[2] == 'base': 
      name_tokens[1:3] = '' 
  elif name == 'squad_trained_newsqa_triviaqa_model_out': 
      name_tokens = 'squad_newsqa_triviaqa_model_out'.split('_') 
  if name_tokens[1] == 'trained': 
      name_tokens[1] = '+' 
  name_tokens[-2:] = '' 
  return '_'.join(name_tokens) 

def read_model_data(model_dir):
  data_dirs = [d for d in os.listdir(model_dir) if d.endswith('_out')]
  data_dirs.sort()
  data = { d: json.load(open(model_dir + '/' + d + '/results.json', 'r'))  for d in data_dirs }
  acc_data = { clean_name(d): data[d]['exact'] for d in data.keys() }
  return acc_data

data = {}
model_data_dirs = ['{}/{}'.format(sys.argv[1], d) for d in os.listdir(sys.argv[1])]
for model_data_dir in model_data_dirs:
  model_data = read_model_data(model_data_dir)
  dataset_name = model_data_dir.split('/')[-1]
  # plot_model_data(dataset_name, model_data)
  data[dataset_name] = model_data

df = pd.DataFrame(data)
print(df)
ax = df.plot(kind='bar')
ax.set_xticklabels(df.index, rotation=20)
plt.yticks(list(range(0, 101, 10)))
plt.grid(axis='y', alpha=0.5)

plt.show()