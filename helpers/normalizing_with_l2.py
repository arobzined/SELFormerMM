#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import normalize

"""EMBEDDING_PATH  = "/home/g3bbmproject/main_folder/train/data"
EMBEDDING_NAMES = ["GRAPH", "TEXT", "KNOWLEDGE_GRAPH"]"""

def main(name: str) -> None:
    embeddings = np.load(name)
    print(f"[{name}] original shape: {embeddings.shape}")

    non_zero_mask = np.linalg.norm(embeddings, axis=1) != 0
    zero_count    = (~non_zero_mask).sum()
    print(f"[{name}] number of ero vectors: {zero_count}")

    processed = embeddings.astype(np.float32, copy=True)

    if non_zero_mask.any():
        mean_vec = processed[non_zero_mask].mean(axis=0, keepdims=True)
        centered = processed[non_zero_mask] - mean_vec

        processed[non_zero_mask] = normalize(centered, norm="l2")

    out_path = f"/home/g3-bbm-project/main_folder/new_data_embeddings/text_embed_LAST_MULTI/text_embeddings_2m_datapoint_normalized.npy"
    np.save(out_path, processed)
    print(f"[{name}] saved → {out_path}")

if __name__ == "__main__":
    main('/home/g3-bbm-project/main_folder/new_data_embeddings/text_embed_LAST_MULTI/text_embeddings_2m_datapoint.npy')
