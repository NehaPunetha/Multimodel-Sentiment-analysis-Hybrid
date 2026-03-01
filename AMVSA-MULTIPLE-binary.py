# ============================================================
# MVSA-Multiple Binary Sentiment (drop neutral)
# Dataset:
#   - Source: extracted_multiple/MVSA
#   - labelResultAll.txt format: id<TAB>text_label,image_label
# Pipeline:
#   1) Build binary CSV:
#        - keep only text_label in {positive, negative}
#        - require text_label == image_label (agreement)
#   2) Train:
#        - Text: fine-tuned roberta-large, num_labels=2
#        - Image: CLIP ViT-B/32 features + linear head (2 classes)
#   3) Evaluate:
#        - text-only
#        - image-only
#        - SOFT fusion
#        - STACKING (LogisticRegressionCV)
#        - MCDM methods (TOPSIS, RAFSI, TODIM, MARCOS, EDAS, SOFT) with per-method best config
# ============================================================

import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
)

# ----------------------------
# Setup
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

LABELS = ["negative", "positive"]
label_map = {"negative": 0, "positive": 1}

SOURCE_DIR = os.path.join("extracted_multiple", "MVSA")
LABEL_FILE = os.path.join(SOURCE_DIR, "labelResultAll.txt")
DATA_DIR   = os.path.join(SOURCE_DIR, "data")
CSV_PATH   = "mvsa_multiple_binary_clean.csv"

DROP_BAD_IMAGES = True
MAX_LEN = 128

TEXT_EPOCHS = 4
IMG_EPOCHS  = 10
TEXT_BS = 16
IMG_BS  = 64
LR_TEXT = 2e-5
LR_IMG  = 8e-4
WD = 1e-4

SOFT_W_GRID = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]

# ----------------------------
# Build binary CSV (agreement + drop neutral)
# ----------------------------
def build_binary_csv():
    if os.path.exists(CSV_PATH):
        print("Using existing:", CSV_PATH)
        return

    if not (os.path.exists(LABEL_FILE) and os.path.isdir(DATA_DIR)):
        raise FileNotFoundError(f"Missing {LABEL_FILE} or {DATA_DIR}")

    rows = []
    with open(LABEL_FILE, "r", encoding="utf-8", errors="ignore") as lf:
        _ = lf.readline()  # header
        for line in lf:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            idx = parts[0].strip()
            lbls = parts[1].split(",")
            text_label = (lbls[0].strip().lower() if len(lbls) > 0 else "")
            img_label  = (lbls[1].strip().lower() if len(lbls) > 1 else text_label)

            # strict agreement + binary only
            if text_label != img_label:
                continue
            if text_label not in label_map:   # skip neutral
                continue

            txt_path = os.path.join(DATA_DIR, f"{idx}.txt")
            if not os.path.exists(txt_path):
                continue
            try:
                text = open(txt_path, "r", encoding="utf-8", errors="ignore").read().strip()
            except Exception:
                continue
            if not text:
                continue

            img_jpg = os.path.join(DATA_DIR, f"{idx}.jpg")
            img_png = os.path.join(DATA_DIR, f"{idx}.png")
            img_path = img_jpg if os.path.exists(img_jpg) else (img_png if os.path.exists(img_png) else "")

            if DROP_BAD_IMAGES:
                if not img_path:
                    continue
                try:
                    _ = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

            rows.append({"id": idx, "text": text, "image": img_path, "label": text_label})

    if not rows:
        raise RuntimeError("No rows after filtering.")
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print("Built:", CSV_PATH, "rows:", len(rows))

build_binary_csv()

# ----------------------------
# Load + split
# ----------------------------
df = pd.read_csv(CSV_PATH).dropna(subset=["text","image","label"]).reset_index(drop=True)
df["label"] = df["label"].astype(str).str.lower()

y_all = np.array([label_map[v] for v in df["label"].values], dtype=np.int64)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=y_all)
train_df, val_df  = train_test_split(
    train_df, test_size=0.1, random_state=SEED,
    stratify=np.array([label_map[v] for v in train_df["label"].values], dtype=np.int64)
)

y_train = np.array([label_map[v] for v in train_df["label"].values], dtype=np.int64)
y_val   = np.array([label_map[v] for v in val_df["label"].values], dtype=np.int64)
y_test  = np.array([label_map[v] for v in test_df["label"].values], dtype=np.int64)

print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

# ============================================================
# TEXT: fine-tune RoBERTa-large (binary)
# ============================================================
tok = RobertaTokenizerFast.from_pretrained("roberta-large")
text_model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large", num_labels=2
).to(device)

class TextDS(Dataset):
    def __init__(self, df_):
        self.texts = df_["text"].astype(str).tolist()
        self.labels = [label_map[v] for v in df_["label"].astype(str).str.lower().tolist()]
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.labels[i]

def collate_text(batch):
    texts, labels = zip(*batch)
    enc = tok(list(texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    return enc, torch.tensor(labels, dtype=torch.long)

def train_text():
    tr_dl = DataLoader(TextDS(train_df), batch_size=TEXT_BS, shuffle=True, collate_fn=collate_text)
    va_dl = DataLoader(TextDS(val_df),   batch_size=TEXT_BS, shuffle=False, collate_fn=collate_text)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(text_model.parameters(), lr=LR_TEXT, weight_decay=WD)

    best = -1.0
    best_state = None
    bad, patience = 0, 2

    for ep in range(1, TEXT_EPOCHS+1):
        text_model.train()
        for enc, yb in tqdm(tr_dl, desc=f"Text ep{ep}"):
            for k in enc: enc[k] = enc[k].to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = text_model(**enc).logits
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)
            opt.step()

        text_model.eval()
        preds = []
        with torch.no_grad():
            for enc, _ in va_dl:
                for k in enc: enc[k] = enc[k].to(device)
                preds.append(torch.argmax(text_model(**enc).logits, dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        va_acc = accuracy_score(y_val, preds)
        print(f"[Text] val_acc={va_acc*100:.2f}%")

        if va_acc > best:
            best = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in text_model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        text_model.load_state_dict(best_state)

@torch.no_grad()
def text_logits(df_):
    dl = DataLoader(TextDS(df_), batch_size=TEXT_BS, shuffle=False, collate_fn=collate_text)
    text_model.eval()
    outs = []
    for enc, _ in tqdm(dl, desc="Text logits"):
        for k in enc: enc[k] = enc[k].to(device)
        outs.append(text_model(**enc).logits.cpu().numpy())
    return np.vstack(outs)

train_text()

# ============================================================
# IMAGE: CLIP ViT-B/32 image + linear binary head
# ============================================================
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip.eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class ImgDS(Dataset):
    def __init__(self, df_):
        self.paths = df_["image"].astype(str).tolist()
        self.labels = [label_map[v] for v in df_["label"].astype(str).str.lower().tolist()]
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.paths[i], self.labels[i]

def collate_img(batch):
    paths, labels = zip(*batch)
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = clip_proc(images=imgs, return_tensors="pt")
    return inputs, torch.tensor(labels, dtype=torch.long)

class CLIPHead(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.2),
            nn.Linear(in_dim, 2)
        )
    def forward(self, x): return self.net(x)

img_head = CLIPHead(512).to(device)

def _clip_feats(inputs):
    feats = clip.get_image_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "image_embeds"):
            feats = feats.image_embeds
        elif isinstance(feats, (tuple, list)):
            feats = feats[0]
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-9)
    return feats

def train_img():
    tr_dl = DataLoader(ImgDS(train_df), batch_size=IMG_BS, shuffle=True, collate_fn=collate_img)
    va_dl = DataLoader(ImgDS(val_df),   batch_size=IMG_BS, shuffle=False, collate_fn=collate_img)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(img_head.parameters(), lr=LR_IMG, weight_decay=WD)

    best = -1.0
    best_state = None
    bad, patience = 0, 3

    for ep in range(1, IMG_EPOCHS+1):
        img_head.train()
        for inputs, yb in tqdm(tr_dl, desc=f"Img ep{ep}"):
            for k in inputs: inputs[k] = inputs[k].to(device)
            yb = yb.to(device)

            with torch.no_grad():
                feats = _clip_feats(inputs)

            opt.zero_grad(set_to_none=True)
            logits = img_head(feats)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(img_head.parameters(), 1.0)
            opt.step()

        img_head.eval()
        preds = []
        with torch.no_grad():
            for inputs, _ in va_dl:
                for k in inputs: inputs[k] = inputs[k].to(device)
                feats = _clip_feats(inputs)
                preds.append(torch.argmax(img_head(feats), dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        va_acc = accuracy_score(y_val, preds)
        print(f"[Image] val_acc={va_acc*100:.2f}%")

        if va_acc > best:
            best = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in img_head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        img_head.load_state_dict(best_state)

@torch.no_grad()
def img_logits(df_):
    dl = DataLoader(ImgDS(df_), batch_size=IMG_BS, shuffle=False, collate_fn=collate_img)
    img_head.eval()
    outs = []
    for inputs, _ in tqdm(dl, desc="Img logits"):
        for k in inputs: inputs[k] = inputs[k].to(device)
        feats = _clip_feats(inputs)
        outs.append(img_head(feats).cpu().numpy())
    return np.vstack(outs)

train_img()

# ============================================================
# Fusion + evaluation (binary)
# ============================================================
def softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-9)

def entropy(P):
    P = np.clip(P, 1e-9, 1.0)
    return -(P * np.log(P)).sum(axis=1, keepdims=True)

def margin(P):
    s = np.sort(P, axis=1)
    return (s[:, -1] - s[:, -2]).reshape(-1, 1)

# logits
tl_va, il_va = text_logits(val_df), img_logits(val_df)
tl_te, il_te = text_logits(test_df), img_logits(test_df)

tp_va, ip_va = softmax_np(tl_va), softmax_np(il_va)
tp_te, ip_te = softmax_np(tl_te), softmax_np(il_te)

print("\nVAL text-only acc:", round(accuracy_score(y_val, np.argmax(tp_va,1))*100,2), "%")
print("VAL img-only acc :", round(accuracy_score(y_val, np.argmax(ip_va,1))*100,2), "%")

# ---- SOFT fusion (tune w on VAL, apply to TEST) ----
best_w, best_acc = None, -1
for w in SOFT_W_GRID:
    pred = np.argmax(w*tp_va + (1.0-w)*ip_va, axis=1)
    acc = accuracy_score(y_val, pred)
    if acc > best_acc:
        best_acc, best_w = acc, w

pred_soft_te = np.argmax(best_w*tp_te + (1.0-best_w)*ip_te, axis=1)
print("\nBest SOFT w on VAL:", best_w, "val_acc:", round(best_acc*100,2), "%")
print("SOFT TEST acc:", round(accuracy_score(y_test, pred_soft_te)*100, 2), "%")
print("SOFT TEST f1 :", round(f1_score(y_test, pred_soft_te, average="binary", zero_division=0)*100, 2), "%")

# ---- STACKING fusion (LogisticRegressionCV) ----
X_meta_va = np.hstack([tl_va, il_va, entropy(tp_va), entropy(ip_va), margin(tp_va), margin(ip_va)]).astype(np.float32)
X_meta_te = np.hstack([tl_te, il_te, entropy(tp_te), entropy(ip_te), margin(tp_te), margin(ip_te)]).astype(np.float32)

sc = StandardScaler()
X_meta_va = sc.fit_transform(X_meta_va)
X_meta_te = sc.transform(X_meta_te)

meta = LogisticRegressionCV(
    Cs=[0.1,0.3,1.0,3.0,10.0],
    cv=5,
    max_iter=6000,
    class_weight="balanced",
    scoring="accuracy"
)
meta.fit(X_meta_va, y_val)
pred_stack_te = meta.predict(X_meta_te)

print("\nSTACKING TEST acc:", round(accuracy_score(y_test, pred_stack_te)*100, 2), "%")
print("STACKING TEST f1 :", round(f1_score(y_test, pred_stack_te, average="binary", zero_division=0)*100, 2), "%")

# ============================================================
# MCDM fusion on binary probs: TOPSIS, RAFSI, TODIM, MARCOS, EDAS, SOFT
# ============================================================
def minmax_norm(D):
    mn = D.min(axis=0); mx = D.max(axis=0)
    denom = np.where((mx-mn)==0, 1.0, (mx-mn))
    return (D - mn) / denom

def topsis(D, W):
    R = D / (np.sqrt((D**2).sum(axis=0)) + 1e-9)
    V = R * W
    ideal = V.max(axis=0); nadir = V.min(axis=0)
    dpos = np.linalg.norm(V - ideal, axis=1)
    dneg = np.linalg.norm(V - nadir, axis=1)
    return dneg / (dpos + dneg + 1e-9)

def rafsi(D, W):
    R = minmax_norm(D)
    ranks = R.argsort(axis=0).argsort(axis=0).astype(np.float32)
    return (ranks * W).sum(axis=1)

def todim(D, W, theta=1.0):
    R = minmax_norm(D)
    n = R.shape[0]
    s = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = R[i] - R[j]
            s[i] += (W * np.maximum(diff, 0)).sum() - theta * (W * np.maximum(-diff, 0)).sum()
    return s

def marcos(D, W):
    R = minmax_norm(D)
    S = R @ W
    return S / (S.sum() + 1e-9)

def edas(D, W):
    AV = D.mean(axis=0)
    PDA = np.maximum(0.0, (D - AV) / (AV + 1e-9))
    NDA = np.maximum(0.0, (AV - D) / (AV + 1e-9))
    SP = (PDA * W).sum(axis=1)
    SN = (NDA * W).sum(axis=1)
    NSP = SP / (SP.max() + 1e-9)
    NSN = 1.0 - (SN / (SN.max() + 1e-9))
    return 0.5 * (NSP + NSN)

MCDM_METHODS = {
    "TOPSIS": lambda D, W, theta: topsis(D, W),
    "RAFSI":  lambda D, W, theta: rafsi(D, W),
    "TODIM":  lambda D, W, theta: todim(D, W, theta=theta),
    "MARCOS": lambda D, W, theta: marcos(D, W),
    "EDAS":   lambda D, W, theta: edas(D, W),
}

def predict_mcdm(method_name, text_probs, img_probs, W, theta=1.0):
    fn = MCDM_METHODS[method_name]
    preds = []
    for i in range(len(text_probs)):
        D = np.stack([text_probs[i], img_probs[i]], axis=1)  # (2,2)
        score = fn(D, W, theta)
        preds.append(int(np.argmax(score)))
    return np.array(preds, dtype=np.int64)

def temp_scale_probs(P, T):
    P = np.clip(P, 1e-9, 1.0)
    logP = np.log(P) / float(T)
    expv = np.exp(logP - logP.max(axis=1, keepdims=True))
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-9)

weight_candidates = [np.array([w, 1.0-w], dtype=np.float32) for w in
                     [0.98,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]]
theta_candidates = [0.25,0.5,0.75,1.0,1.25,1.5,2.0]
temp_candidates  = [0.8, 1.0, 1.2, 1.5, 2.0]

val_results = []

for m in MCDM_METHODS.keys():
    for W in weight_candidates:
        thetas = theta_candidates if m == "TODIM" else [1.0]
        for theta in thetas:
            for T in temp_candidates:
                tp_val = temp_scale_probs(tp_va, T)
                ip_val = temp_scale_probs(ip_va, T)
                pred = predict_mcdm(m, tp_val, ip_val, W=W, theta=theta)
                acc = accuracy_score(y_val, pred)
                f1  = f1_score(y_val, pred, average="binary", zero_division=0)
                val_results.append((acc, f1, m, float(W[0]), float(W[1]), float(theta), float(T)))

# Include SOFT variants in same list for comparison
for tw in [0.98,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]:
    for T in temp_candidates:
        tp_val = temp_scale_probs(tp_va, T)
        ip_val = temp_scale_probs(ip_va, T)
        pred = np.argmax(tw*tp_val + (1.0-tw)*ip_val, axis=1)
        acc = accuracy_score(y_val, pred)
        f1  = f1_score(y_val, pred, average="binary", zero_division=0)
        val_results.append((acc, f1, "SOFT", float(tw), float(1.0-tw), 1.0, float(T)))

val_results.sort(key=lambda x: x[0], reverse=True)

print("\nTop 10 configs on VAL (by Accuracy):")
for i, (acc, f1, m, w0, w1, theta, T) in enumerate(val_results[:10], 1):
    extra = f"theta={theta:.2f}" if m == "TODIM" else ""
    print(f"{i:02d}. {m:6s} val_acc={acc*100:.2f}% val_f1={f1*100:.2f}% W=[{w0:.2f},{w1:.2f}] T={T:.2f} {extra}")

print("\nBest TEST performance per fusion method:")
methods = ["TOPSIS", "RAFSI", "TODIM", "MARCOS", "EDAS", "SOFT"]
for m in methods:
    cand = [r for r in val_results if r[2] == m]
    if not cand:
        continue
    acc, f1, _, w0, w1, theta, T = sorted(cand, key=lambda x: x[0], reverse=True)[0]
    W = np.array([w0, w1], dtype=np.float32)
    tp_test = temp_scale_probs(tp_te, T)
    ip_test = temp_scale_probs(ip_te, T)
    if m == "SOFT":
        test_pred = np.argmax(w0*tp_test + w1*ip_test, axis=1)
    else:
        test_pred = predict_mcdm(m, tp_test, ip_test, W=W, theta=theta)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1  = f1_score(y_test, test_pred, average="binary", zero_division=0)
    print(f"{m:6s} TEST acc={test_acc*100:.2f}% TEST f1={test_f1*100:.2f}% (best VAL_acc={acc*100:.2f}%)")
