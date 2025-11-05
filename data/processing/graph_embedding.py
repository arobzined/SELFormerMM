import os
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime
from unimol_tools import UniMolRepr

unimol_model = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)

def get_unimol_embeddings_batch(smiles_list, model):
    try:
        batch_repr = model.get_repr(smiles_list, return_atomic_reprs=True)
        cls_reprs = batch_repr['cls_repr']
        return np.array(cls_reprs)
    except Exception as e:
        print(f"Error embedding batch: {e}")
        return None

def process_folder_unimol(folder_path, batch_size=2000):
    """
    Walk through the folder and process each CSV file ending with 'filtered'.
    Embeddings are saved in the same folder with '_graph_embedding.npy' added to the filename.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith("filtered.csv") and not file.endswith("mock.csv"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    column_name = 'smiles'
                    if column_name not in df.columns:
                        column_name = 'mol'

                        if column_name not in df.columns:
                            raise ValueError("'smiles' column not found in the CSV file.")

                    df = df.dropna(subset=[column_name])
                    smiles_list = df[column_name].tolist()
                    print(f"Found {len(smiles_list)} valid SMILES to process.")

                    all_embeddings = []
                    for i in range(0, len(smiles_list), batch_size):
                        batch = smiles_list[i:i+batch_size]
                        embeddings = get_unimol_embeddings_batch(batch, unimol_model)
                        if embeddings is not None:
                            all_embeddings.append(embeddings)
                        else:
                            print(f"Warning: Batch {i//batch_size} failed.")

                    if all_embeddings:
                        final_embeddings = np.concatenate(all_embeddings)
                        output_file = os.path.join(root, f"{os.path.splitext(file)[0]}_graph_embedding.npy")
                        np.save(output_file, final_embeddings)
                        print(f"Saved embeddings with shape {final_embeddings.shape} to {output_file}\n")
                    else:
                        print(f"No embeddings generated for {file_path}.")

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}\n")

folder_path = "/home/g3-bbm-project/main_folder/FineTune/finetune_data_multi/finetuning_datasets/classification"  # Set your top-level folder here
print(f"Starting UniMol embedding processing at {datetime.now().strftime('%H:%M:%S')}")
start_time = time.time()

process_folder_unimol(folder_path)

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")
print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")