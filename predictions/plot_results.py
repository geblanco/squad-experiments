from matplotlib import pyplot as plt
from collections import defaultdict
import json, os

data_dirs = [d for d in os.listdir('./') if d.endswith('_data')]
data_dirs.sort()
data = { d: json.load(open(d + '/results.json', 'r'))  for d in data_dirs }
acc_data = { d: data[d]['exact'] for d in data.keys() }

# get two views to data, indexed by model and by dataset
datasets = ['NewsQA', 'TriviaQA', 'SQuAD']
models = ['base', 'mixed', 'newsqa', 'triviaqa']
datasets_dict = defaultdict(list)
models_dict = defaultdict(list)

for name, value in acc_data.items():
  for model in models:
    if name.startswith(model):
      models_dict[model].append(value)
  for dataset in datasets:
    dataset_name = dataset.lower()
    if name.split('_')[1] == dataset_name:
      datasets_dict[dataset].append(value)

fig, ax = plt.subplots(figsize=(10,5))
x = list(range(len(datasets_dict[datasets[0]])))
width = 0.20

for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
  print(dataset_name, dataset)
  # subtract width to center bars so legend gets center-aligned
  positions = [(p + (width * i)) - width for p in x]
  plt.bar(positions, dataset, width, alpha=0.75, label=dataset_name)

ax.set_ylabel('Exact Matches')
#plt.ylim([0, 100])
plt.yticks(list(range(0, 101, 10)))
plt.xticks([], [])

plt.legend(datasets_dict.keys(), loc='upper left')
ax.set_xticks(x) 
ax.set_xticklabels([m.capitalize() + ' Model' for m in models])
print(models)
plt.grid(axis='y', alpha=0.5)
plt.show()
