from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os

FLAGS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embeddings_dir', '-d', type=str, required=True,
      help='dir with question and context embeddings')
  parser.add_argument('--output_dir', type=str, required=False, 
      help='directory to drop the processed data')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def load_embeddings(embeddings_dir):
  embeddings = []
  embedding_batches = [dir for dir in os.listdir(embeddings_dir) 
      if dir.endswith('.npy')]
  for batch in embedding_batches:
    embedding_batch = np.load(os.path.join(embeddings_dir, batch),
        allow_pickle=True)
    embeddings = np.append(embeddings, embedding_batch)

  return embeddings

def decompose(embeddings):
  pca = PCA(n_components=30)
  pca.fit(embeddings)
  return pca, pca.transform(embeddings)

def get_representation(embeddings):
  tsne = TSNE()
  tsne.fit(embeddings)
  return tsne, tsne.transform(embeddings)

def main():
  embedded_data = load_embeddings(FLAGS.embeddings_dir)
  decomposer, decomp_embeddings = decompose(embedded_data)
  _, repr_embeddings = get_representation(decomp_embeddings)
  repr_df = pd.DataFrame(repr_embeddings)
  repr_df.columns = ['x', 'y']
  sns.lmplot(x='x', y='y', data=repr_df)
  plt.show()

if __name__ == '__main__':
  FLAGS = parse_args()
  main()

