# -----------------------
# Environment & Imports
# -----------------------
import os, sys, argparse
import torch, torch.nn as nn
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaTokenizerFast
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb

# Ortam ayarları
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZER_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

sys.path.append("/home/g3-bbm-project/main_folder")
from multimodal_roberta import MultimodalRoberta
from train.finetuning.prepare_finetuning_data import smiles_to_selfies

# -----------------------
# Argument Parser
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lr",     type=float, default=2e-5, help="Öğrenme hızı")
parser.add_argument("--epochs", type=int,   default=100,   help="Epoch sayısı")
args = parser.parse_args()

# wandb başlat
wandb.init(
    project="pdbbind_regression",
    name=f"lr={args.lr}_epochs={args.epochs}",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": 32,
        "model": "MultimodalRoberta + RegressionHead",
        "dataset": "PDBBind_Full"
    }
)

# -----------------------
# Yol Tanımları
# -----------------------
model_path   = "/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/hf"
weights_path = "/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/model.safetensors"

dataset_path         = "/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/regression/annotated_results/pdbbind_full/pdbbind_full_annotated.csv"
graph_embeddings_path = "/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/regression/annotated_results/pdbbind_full/pdbbind_full_annotated_graph_embedding.npy"
text_embeddings_path  = "/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/regression/annotated_results/pdbbind_full/pdbbind_full_text_embedding.npy"
kg_embeddings_path    = "/home/g3-bbm-project/main_folder/FineTune/dataset_dummy/kg_embeddings_multilabel.npy"

save_to = "/home/g3-bbm-project/main_folder/FineTune/regressionoutputs/pdbbind_full_regression"

# -----------------------
# Config & Tokenizer
# -----------------------
config    = RobertaConfig.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

# -----------------------
# Pretrained Model Load
# -----------------------
pretrained_model = MultimodalRoberta(config)
state = load_file(weights_path)
missing, unexpected = pretrained_model.load_state_dict(state, strict=False)
print(f"🚩 missing params: {len(missing)} | unexpected params: {len(unexpected)}")

for name, param in pretrained_model.named_parameters():
    if any(f"encoder.layer.{i}" in name for i in [9, 10, 11]) or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# -----------------------
# Regression Head
# -----------------------
class MultimodalRegressionModel(nn.Module):
    def __init__(self, backbone, hidden_size=768, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, graph_emb, text_emb, kg_emb, input_ids, attention_mask, labels=None):
        embeds = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_emb=graph_emb,
            text_emb=text_emb,
            kg_emb=kg_emb
        )
        B = input_ids.size(0)
        combined = torch.cat([embeds[0:B], embeds[B:2*B], embeds[2*B:3*B], embeds[3*B:4*B]], dim=1)
        preds = self.regressor(combined)
        loss = nn.MSELoss()(preds.view(-1), labels.view(-1)) if labels is not None else None
        return loss, preds

# -----------------------
# Dataset
# -----------------------
class CustomRegressionDataset(Dataset):
    def __init__(self, df, graph_emb, text_emb, kg_emb, tokenizer, max_len=512):
        self.df, self.g, self.t, self.k = df.reset_index(drop=True), graph_emb, text_emb, kg_emb
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx, 0]
        label  = self.df.iloc[idx, 1]
        enc = self.tok.encode_plus(
            smiles,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "graph_emb": torch.tensor(self.g[idx], dtype=torch.float32),
            "text_emb":  torch.tensor(self.t[idx], dtype=torch.float32),
            "kg_emb":    torch.tensor(self.k[idx], dtype=torch.float32),
            "labels":    torch.tensor(label,   dtype=torch.float32)
        }

# -----------------------
# Data Preparation
# -----------------------
df  = pd.read_csv(dataset_path)
g   = np.load(graph_embeddings_path)
t   = np.load(text_embeddings_path)
kg  = np.load(kg_embeddings_path)[:len(df)]

assert len(df) == len(g) == len(t) == len(kg), "CSV ve embedding boyutları uyuşmuyor!"

train_df, val_df, g_tr, g_val, t_tr, t_val, kg_tr, kg_val = train_test_split(df, g, t, kg, test_size=0.1, random_state=42)

train_ds = CustomRegressionDataset(train_df, g_tr, t_tr, kg_tr, tokenizer)
val_ds   = CustomRegressionDataset(val_df,   g_val, t_val, kg_val, tokenizer)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

# -----------------------
# Model & Optimizer
# -----------------------
model     = MultimodalRegressionModel(pretrained_model)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# -----------------------
# Evaluation Functions
# -----------------------
def evaluate(model, loader):
    model.eval()
    preds, trues, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            g, t, kg  = batch["graph_emb"].to(device), batch["text_emb"].to(device), batch["kg_emb"].to(device)
            y         = batch["labels"].to(device)
            loss, out = model(g, t, kg, ids, mask, y)
            total_loss += loss.item()
            preds.append(out.cpu())
            trues.append(y.cpu())
    preds = torch.cat(preds); trues = torch.cat(trues)
    mse  = mean_squared_error(trues.numpy(), preds.numpy(), squared=True)
    rmse = mean_squared_error(trues.numpy(), preds.numpy(), squared=False)
    mae  = mean_absolute_error(trues.numpy(), preds.numpy())
    return total_loss / len(loader), mse, rmse, mae

train_evaluate = evaluate  # aynı işlemi yapıyor

# -----------------------
# Training Loop
# -----------------------
print("\n🚀 Eğitim başlıyor...\n")
for ep in range(1, args.epochs + 1):
    model.train()
    running = 0
    for batch in train_loader:
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        g, t, kg  = batch["graph_emb"].to(device), batch["text_emb"].to(device), batch["kg_emb"].to(device)
        y         = batch["labels"].to(device)
        optim.zero_grad()
        loss, _ = model(g, t, kg, ids, mask, y)
        loss.backward()
        optim.step()
        running += loss.item()

    tr_loss = running / len(train_loader)
    tr_eval_loss, tr_mse, tr_rmse, tr_mae = train_evaluate(model, train_loader)
    val_loss, val_mse, val_rmse, val_mae = evaluate(model, val_loader)

    print(f"\nEpoch {ep:02d}")
    print(f"  📊 Train     -> Loss: {tr_eval_loss:.4f} | MSE: {tr_mse:.4f} | RMSE: {tr_rmse:.4f} | MAE: {tr_mae:.4f}")
    print(f"  🧪 Validation -> Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")

    wandb.log({
        "epoch": ep,
        "train_loss": tr_eval_loss,
        "train_mse": tr_mse,
        "train_rmse": tr_rmse,
        "train_mae": tr_mae,
        "val_loss": val_loss,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_mae": val_mae
    })

# -----------------------
# Save Model & Tokenizer
# -----------------------
save_dir = f"/home/g3-bbm-project/main_folder/FineTune/classification_codes/saved_models_LAST/PDBBIND_FULL"
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, "multimodal_PDBBIND_FULL_classifier_sd.pt")

torch.save({
    "state_dict": model.state_dict(),
    "config": config.to_dict()
}, checkpoint_path)

print("✅ Model kaydedildi:", checkpoint_path)

tokenizer.save_pretrained(save_dir)
print("✅ Tokenizer da kaydedildi:", save_dir)
