import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SciBERT tokenizer and model
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
scibert_model.eval()  # Set the model to evaluation mode

# Function to generate text embeddings for a single text
def get_text_embeddings(text, tokenizer, model, device):
    if isinstance(text, str) and text.strip() != "":
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(tokens)
        text_out = output[0][0].mean(dim=0)
    else:
        text_out = torch.zeros(768).to(device)
    return text_out.cpu().numpy()

# New function: Process all CSV files ending with 'filtered' in a folder and its subfolders
def process_folder(folder_path):
    """
    Walk through the folder and process each CSV file ending with 'filtered'.
    Embeddings are saved in the same folder with '_text_embedding.npy' added to the base filename.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("filtered.csv"):
                filtered_path = os.path.join(root, file)
                base_filename = file.replace("_filtered", "")
                full_path = os.path.join(root, base_filename)

                if not os.path.exists(full_path):
                    print(f"Corresponding full CSV file not found for {filtered_path}, skipping.\n")
                    continue

                print(f"Processing file: {filtered_path}")

                try:
                    # Read both full and filtered CSV files
                    df_full = pd.read_csv(full_path)
                    df_filtered = pd.read_csv(filtered_path)

                    '''if ("smiles" not in df_full.columns or "smiles" not in df_filtered.column):
                        raise ValueError("'smiles' column not found in one of the CSV files.")

                    if "Description" not in df_filtered.columns:
                        raise ValueError("'Description' column not found in filtered CSV file.")'''

                    # Map smiles to description

                    column_name = 'smiles'
                    if base_filename == 'bace':
                        column_name = 'mol'

                    smiles_to_description = dict(zip(df_filtered["smiles"], df_filtered["Description"]))

                    # Prepare Description column for full df (may include Nones)
                    df_full["Description"] = df_full[column_name].map(smiles_to_description)

                    # Now generate embeddings
                    tqdm.pandas(desc=f"Embedding {base_filename}")
                    embeddings = df_full["Description"].progress_apply(
                        lambda text: get_text_embeddings(text, scibert_tokenizer, scibert_model, device)
                    ).tolist()

                    embeddings_array = np.array(embeddings)

                    output_file = os.path.join(root, f"{os.path.splitext(base_filename)[0]}_text_embedding.npy")
                    np.save(output_file, embeddings_array)

                    print(f"Saved embeddings to {output_file}\n")

                except Exception as e:
                    print(f"Failed to process {filtered_path}: {e}\n")

# Example usage
folder_path = "/home/g3-bbm-project/main_folder/FineTune/finetune_data_multi/finetuning_datasets/classification"  # Set your top-level folder here
print(f"Starting to process folder: {folder_path}")
process_folder(folder_path)
print("Folder processing complete.")
