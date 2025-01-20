import os
import pandas as pd
from pandarallel import pandarallel
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


SELFIES_DATASET_PATH = "data/temp_selfies.csv"  			# path to the SELFIES dataset
MODEL_FILE_PATH = "data/pretrained_models/SELFormer"       	# path to the pre-trained SELFormer model
OUTPUT_EMBEDDINGS_PATH = "data/embeddings.csv"             	# path to save the generated embeddings


df = pd.read_csv(SELFIES_DATASET_PATH)						# load the dataset
print(f"Loaded dataset with {len(df)} molecules.")


config = RobertaConfig.from_pretrained(MODEL_FILE_PATH)		# load the pre-trained model and tokenizer
config.output_hidden_states = True
tokenizer = RobertaTokenizer.from_pretrained("data/RobertaFastTokenizer")
model = RobertaModel.from_pretrained(MODEL_FILE_PATH, config=config)


def get_sequence_embeddings(selfies):
    token = torch.tensor([tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)])	# tokenize the SELFIES string
    output = model(token)																										# forward pass through the model
    sequence_out = output[0]																									# extract the sequence output and compute the mean pooling
    return torch.mean(sequence_out[0], dim=0).tolist()

print("Generating embeddings...")
pandarallel.initialize(nb_workers=5, progress_bar=True)
df["sequence_embeddings"] = df.selfies.parallel_apply(get_sequence_embeddings)

df.drop(columns=["selfies"], inplace=True)
df.to_csv(OUTPUT_EMBEDDINGS_PATH, index=False)
print(f"Embeddings saved to {OUTPUT_EMBEDDINGS_PATH}")
