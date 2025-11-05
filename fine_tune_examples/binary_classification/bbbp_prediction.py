#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, sys, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaTokenizerFast

sys.path.append("/home/g3-bbm-project/main_folder")
from train.finetuning.prepare_finetuning_data import smiles_to_selfies

from multimodal_roberta import MultimodalRoberta
from bbbp_classification_arc import MultimodalClassificationModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  required=True, help=".pt checkpoint")
parser.add_argument("--tokenizer_path", required=True, help="Tokenizer klasörü")
g = parser.add_mutually_exclusive_group(required=True)
g.add_argument("--input_file", help="CSV (smiles sütunu)")
g.add_argument("--smiles", help="Tek SMILES string")
parser.add_argument("--output_dir", default="predictions")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len",   type=int, default=512)
parser.add_argument("--graph_dim", type=int, default=512)
parser.add_argument("--text_dim",  type=int, default=768)
parser.add_argument("--kg_dim",    type=int, default=128)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path,
                                                 do_lower_case=False)

ckpt = torch.load(args.model_path, map_location=device)

if isinstance(ckpt, dict) and "state_dict" in ckpt: 
    config = RobertaConfig.from_dict(ckpt["config"])
    base   = MultimodalRoberta(config)
    model  = MultimodalClassificationModel(base)
    model.load_state_dict(ckpt["state_dict"])
else:
    model = ckpt
model.eval().to(device)

hidden_size = model.pretrained_model.hidden_size \
              if hasattr(model.pretrained_model, "hidden_size") \
              else config.hidden_size

graph_dim = args.graph_dim or hidden_size
text_dim  = args.text_dim  or hidden_size
kg_dim    = args.kg_dim

if args.input_file:
    df = pd.read_csv(args.input_file)
    if "smiles" not in df.columns:
        raise ValueError("CSV içinde 'smiles' adlı sütun bulunmadı.")
    smiles_list = df["smiles"].astype(str).tolist()
else:
    smiles_list = [args.smiles]
    df = pd.DataFrame({"smiles": smiles_list})

selfies = smiles_to_selfies(df)["selfies"].tolist()

class InferDS(Dataset):
    def __init__(self, selfies, tok, max_len, gd, td, kd):
        self.s = selfies; self.tok = tok; self.mlen = max_len
        self.gd, self.td, self.kd = gd, td, kd
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        enc = self.tok.encode_plus(self.s[i], max_length=self.mlen,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "graph_emb": torch.zeros(self.gd),
            "text_emb":  torch.zeros(self.td),
            "kg_emb":    torch.zeros(self.kd)
        }

ds = InferDS(selfies, tokenizer, args.max_len, graph_dim, text_dim, kg_dim)
dl = DataLoader(ds, batch_size=args.batch_size)

all_emb, all_pred = [], []
for batch in dl:
    batch = {k: v.to(device) for k, v in batch.items()}

    em = model.pretrained_model(**{k: batch[k] for k in
           ("input_ids", "attention_mask", "graph_emb", "text_emb", "kg_emb")})
    B = batch["input_ids"].size(0)
    comb = torch.cat([em[0:B], em[B:2*B], em[2*B:3*B], em[3*B:4*B]], 1)
    all_emb.append(comb.cpu())

    _, lg = model(**batch, labels=None)
    pr = torch.sigmoid(lg.squeeze()).cpu()
    all_pred.append((pr > 0.5).long())

embeddings  = torch.cat(all_emb).numpy().astype("float32")
predictions = torch.cat(all_pred).numpy()

os.makedirs(args.output_dir, exist_ok=True)

out_csv = os.path.join(args.output_dir, "predictions.csv")
df_out = df.copy(); df_out["prediction"] = predictions
df_out.to_csv(out_csv, index=False)

out_npy = os.path.join(args.output_dir, "embeddings.npy")
np.save(out_npy, embeddings)