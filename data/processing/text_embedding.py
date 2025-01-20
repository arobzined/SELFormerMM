import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")       # load SciBERT tokenizer and model
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
scibert_model.eval()                                                                        # set the model to evaluation mode


def get_text_embeddings(text, tokenizer, model, device):
    """
    Generates text embeddings for a single text using SciBERT with GPU acceleration.
    """
    if isinstance(text, str):
        tokens = tokenizer.encode(
            text, add_special_tokens=True, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(tokens)
        text_out = output[0][0].mean(dim=0)             # mean pooling for sequence embeddings
    else:
        text_out = torch.zeros(768).to(device)          # default embedding for non-string inputs
    return text_out.cpu().numpy()                       # convert to NumPy array

def process_csv_to_npy(file_path, output_path):
    """
    Processes all rows in a CSV file to generate embeddings and saves as a NumPy file.
    """
    data = pd.read_csv(file_path)
    if "Description" not in data.columns:
        raise ValueError("'Description' column not found in the CSV file.")
    
    
    tqdm.pandas(desc="Generating Embeddings")           # generate embeddings for each row in the 'Description' column
    embeddings = data["Description"].fillna("").progress_apply(
        lambda text: get_text_embeddings(text, scibert_tokenizer, scibert_model, device)
    ).tolist()
    
    embeddings_array = np.array(embeddings)
    np.save(output_path, embeddings_array)
    print(f"Processed all rows and saved embeddings to {output_path}")

input_file = "/home/g3bbmproject/main_folder/KG/kg.pt/final_matched_data_with_embeddings.csv"
output_file = "embeddings_of_text_using_kg_embedding_data.npy"

print(f"Processing file: {input_file}")
process_csv_to_npy(input_file, output_file)
print(f"Embedding generation complete. Results saved to {output_file}")