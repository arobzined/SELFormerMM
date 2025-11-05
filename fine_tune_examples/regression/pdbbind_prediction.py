#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, sys, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaTokenizerFast
from typing import Dict, List

sys.path.extend([
    "/home/g3-bbm-project/main_folder",
    "/home/g3-bbm-project/main_folder/FineTune/classification_codes",
])
from multimodal_roberta import MultimodalRoberta
from pdbbind_regression_arc import MultimodalRegressionModel

p = argparse.ArgumentParser()
p.add_argument("--model_path",     required=True,
               help=".pt checkpoint (state_dict + config)")
p.add_argument("--tokenizer_path", required=True,
               help="HF tokenizer klasörü (epoch_XXX/hf)")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--input_file", help="CSV (smiles sütunu)")
g.add_argument("--smiles",     help="Tek SMILES stringi")
p.add_argument("--graph_embs", help="Opsiyonel .npy dosyası (N × H)")
p.add_argument("--text_embs",  help="Opsiyonel .npy dosyası (N × H)")
p.add_argument("--kg_embs",    help="Opsiyonel .npy dosyası (N × K)")
p.add_argument("--output_dir", default="predictions")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--max_len",    type=int, default=512)
args = p.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

ckpt = torch.load(args.model_path, map_location="cpu")
if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "config" in ckpt):
    raise ValueError("Checkpoint, {'state_dict','config'} içermiyor.")

state_dict = ckpt["state_dict"]
config     = RobertaConfig.from_dict(ckpt["config"])
tokenizer  = RobertaTokenizerFast.from_pretrained(args.tokenizer_path,
                                                  do_lower_case=False)

backbone = MultimodalRoberta(config)
model    = MultimodalRegressionModel(backbone)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:    
    print(f"⚠️ Missing parameters: {len(missing)}")
if unexpected: 
    print(f"⚠️ Unexpected parameters {len(unexpected)}")

model.eval().to(DEVICE)

if args.input_file:
    df = pd.read_csv(args.input_file)
    if "smiles" not in df.columns:
        raise ValueError("No'smiles' column in CSV.")
    smiles = df["smiles"].astype(str).tolist()
else:
    smiles = [args.smiles]
    df = pd.DataFrame({"smiles": smiles})

N = len(df)
def load_opt(path, name, dim):
    if path:
        arr = np.load(path)
        if arr.shape[0] < N:
            raise ValueError(f"{name} satır sayısı CSV'den az.")
        return arr[:N]
    return np.zeros((N, dim), dtype="float32")

graph = load_opt(args.graph_embs, "graph_embs", 512)
text  = load_opt(args.text_embs,  "text_embs",  768)
kg    = load_opt(args.kg_embs,    "kg_embs",    128)

class InferDS(Dataset):
    def __init__(self, smiles: List[str]):
        self.s = smiles
    def __len__(self): return len(self.s)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        enc = tokenizer(self.s[i], max_length=args.max_len,
                        truncation=True, padding="max_length",
                        return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "graph_emb": torch.tensor(graph[i]),
            "text_emb":  torch.tensor(text[i]),
            "kg_emb":    torch.tensor(kg[i]),
        }

dl = DataLoader(InferDS(smiles), batch_size=args.batch_size,
                pin_memory=torch.cuda.is_available())

all_preds, all_embs = [], []
with torch.no_grad():
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out, emb = model(**batch)
        all_preds.append(out.squeeze(1).cpu())
        all_embs.append(emb.cpu())

preds = torch.cat(all_preds).numpy()
embs  = torch.cat(all_embs).numpy().astype("float32")

os.makedirs(args.output_dir, exist_ok=True)

df_out = df.copy()
df_out["prediction"] = preds
csv_path = os.path.join(args.output_dir, "regression_predictions.csv")
df_out.to_csv(csv_path, index=False)

emb_path = os.path.join(args.output_dir, "embeddings.npy")
np.save(emb_path, embs)