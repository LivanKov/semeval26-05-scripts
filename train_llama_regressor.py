#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


# -------------------------
# Dataset
# -------------------------
class JsonlRegressionDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append({"text": data["text"], "label": float(data["label"])})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -------------------------
# Regressor head
# -------------------------
class Regressor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------
# Mean pooling
# -------------------------
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(embedder, regressor, loader, device):
    embedder.eval(); regressor.eval()
    preds, trues = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = embedder(input_ids=input_ids, attention_mask=attention_mask)
        embeds = mean_pooling(outputs.last_hidden_state, attention_mask)
        pred = regressor(embeds)

        preds.extend(pred.cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())

    mse = mean_squared_error(trues, preds)
    spearman = spearmanr(trues, preds).correlation
    return mse, spearman


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="unsloth/llama-3.1-8b-instruct-bnb-4bit")  # ← UNGATED + 4bit
    parser.add_argument("--output_dir", type=str, default="models/llama_regressor_unsloth")
    parser.add_argument("--batch_size", type=int, default=28)       # ← fits easily now
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model (already 4-bit quantized → no extra flags needed)
    embedder = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False

    # Dataset
    dataset = JsonlRegressionDataset(args.train_jsonl)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        labels = [x["label"] for x in batch]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32)
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Get embedding dim
    sample = next(iter(train_loader))
    with torch.no_grad():
        dim = embedder(**{k: v.to(embedder.device) for k, v in sample.items() if k != "labels"}).last_hidden_state.shape[-1]

    # Regressor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regressor = Regressor(dim).to(device)
    optimizer = torch.optim.AdamW(regressor.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_spearman = -1.0
    print(f"Training on {len(train_ds)} samples | Val: {len(val_ds)} | Embedding dim: {dim}")

    for epoch in range(1, args.epochs + 1):
        regressor.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with torch.no_grad():
                out = embedder(**batch)
                embeds = mean_pooling(out.last_hidden_state, batch["attention_mask"])

            preds = regressor(embeds)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_mse, val_spearman = evaluate(embedder, regressor, val_loader, device)
        print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f} | Val MSE: {val_mse:.4f} | Spearman: {val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save(regressor.state_dict(), os.path.join(args.output_dir, "regressor_best.pt"))
            json.dump({"model_name": args.model_name, "spearman": best_spearman}, 
                      open(os.path.join(args.output_dir, "meta.json"), "w"))
            print(f"NEW BEST → Spearman = {best_spearman:.4f}")

    print("Done! Best Spearman:", best_spearman)


if __name__ == "__main__":
    main()