#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import normalize

EMBEDDING_PATH = '/home/g3-bbm-project/main_folder/train/2milyon_normalized_embeddings'
EMBEDDING_NAME = 'graph_embeddings_2m'

def main(name):
    full_path = f"{EMBEDDING_PATH}/{name}.npy"

    embeddings = np.load(full_path)
    print('Shape of the dataset:', embeddings.shape)

    mean_vec = np.mean(embeddings, axis=0, keepdims=True)
    embeddings_centered = embeddings - mean_vec

    embeddings_normalized = normalize(embeddings_centered, norm='l2')

    output_path = f"{EMBEDDING_PATH}/{name}_normalized.npy"
    np.save(output_path, embeddings_normalized)
    print(f"Saved normalized embeddings to {output_path}")

if __name__ == "__main__":
    main(EMBEDDING_NAME)
