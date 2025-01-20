import pandas as pd
import numpy as np
from unimol_tools import UniMolRepr
import torch

# Initialize UniMol Model with GPU support
unimol_model = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)

# Define the function to generate UniMol embeddings
def get_unimol_embeddings(smiles, model):
    """
    Generate UniMol embeddings for a given SMILES string.
    :param smiles: A SMILES string representing a molecule.
    :param model: UniMolRepr model instance.
    :return: UniMol CLS token representation as a list.
    """
    try:
        unimol_repr = model.get_repr(smiles, return_atomic_reprs=True)  # Generate embeddings
        cls_repr = unimol_repr['cls_repr']  # CLS token embedding (molecular representation)
        return np.array(cls_repr)
    except Exception as e:
        print(f"Error embedding SMILES {smiles}: {e}")
        return None  # Return None if an error occurs

# Load the CSV file
input_file = "/home/g3bbmproject/main_folder/KG/kg.pt/final_matched_data_with_embeddings.csv"
df = pd.read_csv(input_file)

# Ensure the 'smiles' column exists
if 'smiles' not in df.columns:
    raise ValueError("The input file does not contain a 'smiles' column.")

# Define chunk size
chunk_size = 100000  # Process 100,000 rows at a time

# Create a generator for chunking the data
def chunk_data(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

# Process each chunk
output_file_prefix = "/home/g3bbmproject/main_folder/M^3-Datasets-20241215T191503Z-001/M^3-Datasets/graph_embed_data/new_graph_embeddings_based_on_kg_erva"

total_chunks = (len(df) // chunk_size) + 1
print(f"Total chunks to process: {total_chunks}")

# Verify GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for idx, chunk in enumerate(chunk_data(df, chunk_size)):
    print(f"Processing chunk {idx + 1} of {total_chunks}...")

    # Generate embeddings
    chunk['unimol_embeddings'] = chunk['smiles'].apply(lambda x: get_unimol_embeddings(x, unimol_model))

    # Drop rows where embeddings failed
    chunk = chunk[chunk['unimol_embeddings'].notnull()]

    # Save the chunk as a NumPy file
    chunk_output_file = f"{output_file_prefix}_chunk_{idx + 1}.npy"
    embeddings_array = np.array(chunk['unimol_embeddings'].tolist())
    np.save(chunk_output_file, embeddings_array)
    print(f"Chunk {idx + 1} saved to {chunk_output_file}")

    # Log progress
    print(f"Completed {((idx + 1) / total_chunks) * 100:.2f}% of processing.")

print("All chunks processed and saved.")
