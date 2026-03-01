# ============================================================
# MVSA Binary Sentiment (drop neutral)
# Pipeline:
# 1) Build clean CSV from MVSA_Single labelResultAll.txt
#    - keep only agreement: text_label == image_label
#    - keep only labels in {negative, positive}
# 2) Train frozen RoBERTa-large embeddings + TextMLP (2 classes)
# 3) Train frozen VGG16 features + ImageMLP (2 classes)
# 4) Evaluate:
#    - text-only, image-only
#    - logits-based STACKING fusion (meta LogisticRegression trained on VAL)
#    - MCDM grid search: TOPSIS, RAFSI, TODIM, MARCOS, EDAS + SOFT (tuned on VAL)
# ============================================================

import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms

from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Setup
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Binary labels (neutral dropped)
LABELS = ["negative", "positive"]
label_map = {"negative": 0, "positive": 1}

# MVSA_Single extracted paths
SOURCE_DIR = os.path.join("extracted_data", "MVSA_Single")
LABEL_FILE = os.path.join(SOURCE_DIR, "labelResultAll.txt")
DATA_DIR   = os.path.join(SOURCE_DIR, "data")

DATA_PATH  = "mvsa_binary_clean.csv"
DROP_BAD_IMAGES = True
MAX_LEN = 128

# ----------------------------
# Build binary clean dataset CSV
# Keep only: text_label == image_label AND label in {neg,pos}
# ----------------------------
def build_binary_clean_csv():
    if os.path.exists(DATA_PATH):
        print(f"Using existing dataset: {DATA_PATH}")
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
            if text_label not in label_map:
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

            rows.append({"text": text, "image": img_path, "label": text_label})

    if not rows:
        raise RuntimeError("No rows after filtering. Check paths or relax filters.")
    pd.DataFrame(rows).to_csv(DATA_PATH, index=False)
    print(f"Built {DATA_PATH} with {len(rows)} rows")

build_binary_clean_csv()

# ----------------------------
# Load data
# ----------------------------
data = pd.read_csv(DATA_PATH).dropna(subset=["text", "image", "label"]).reset_index(drop=True)
data["label"] = data["label"].astype(str).str.lower()

bad = set(data["label"].unique()) - set(label_map.keys())
if bad:
    raise ValueError(f"Unexpected labels: {bad}")

texts = data["text"].astype(str).values
images = data["image"].astype(str).values
y = np.array([label_map[v] for v in data["label"].values], dtype=np.int64)

# ----------------------------
# Split
# ----------------------------
train_text, test_text, train_img, test_img, y_train, y_test = train_test_split(
    texts, images, y, test_size=0.2, random_state=SEED, stratify=y
)
train_text, val_text, train_img, val_img, y_train, y_val = train_test_split(
    train_text, train_img, y_train, test_size=0.1, random_state=SEED, stratify=y_train
)

print("Train:", len(y_train), "Val:", len(y_val), "Test:", len(y_test))
print("Class counts:", dict(zip(LABELS, np.bincount(y, minlength=2))))

# ============================================================
# Text embeddings (RoBERTa-large frozen)
# ============================================================
print("Loading RoBERTa-large...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
roberta = RobertaModel.from_pretrained("roberta-large").to(device)
roberta.eval()

@torch.no_grad()
def get_text_embedding(text: str) -> np.ndarray:
    inp = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding="max_length", max_length=MAX_LEN
    ).to(device)
    out = roberta(**inp)
    h = out.last_hidden_state  # (1,L,1024)
    cls = h[:, 0, :]
    mean = h.mean(dim=1)
    maxp = h.max(dim=1).values
    emb = torch.cat([cls, mean, maxp], dim=1)  # (1,3072)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

def embed_texts(arr):
    feats = []
    for t in tqdm(arr, desc="Text embeddings"):
        feats.append(get_text_embedding(str(t)))
    return np.vstack(feats)

print("Extracting text features...")
X_text_tr = embed_texts(train_text)
X_text_va = embed_texts(val_text)
X_text_te = embed_texts(test_text)

# ============================================================
# Image features (VGG16 frozen -> 25088)
# ============================================================
print("Loading VGG16...")
try:
    vgg = models.vgg16(weights=models.VGG16Weights.IMAGENET1K_V1).to(device)
except Exception:
    vgg = models.vgg16(pretrained=True).to(device)
vgg.eval()

img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@torch.no_grad()
def get_vgg_feat(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    x = img_tf(img).unsqueeze(0).to(device)
    feat = vgg.features(x)
    feat = vgg.avgpool(feat)
    feat = torch.flatten(feat, 1)  # (1,25088)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

def embed_images(paths):
    feats = []
    for p in tqdm(paths, desc="Image features"):
        feats.append(get_vgg_feat(p))
    return np.vstack(feats)

print("Extracting image features...")
X_img_tr = embed_images(train_img)
X_img_va = embed_images(val_img)
X_img_te = embed_images(test_img)

# ============================================================
# Scale
# ============================================================
text_scaler = StandardScaler()
img_scaler  = StandardScaler()

X_text_tr = text_scaler.fit_transform(X_text_tr)
X_text_va = text_scaler.transform(X_text_va)
X_text_te = text_scaler.transform(X_text_te)

X_img_tr = img_scaler.fit_transform(X_img_tr)
X_img_va = img_scaler.transform(X_img_va)
X_img_te = img_scaler.transform(X_img_te)

# ============================================================
# Models (binary)
# ============================================================
class TextMLP(nn.Module):
    def __init__(self, in_dim=3072, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

class ImageMLP(nn.Module):
    def __init__(self, in_dim=25088, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        logp = F.log_softmax(logits, dim=1)
        ce = -(true_dist * logp).sum(dim=1)
        if self.weight is not None:
            ce = ce * self.weight[targets]
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def train_classifier(model, Xtr, ytr, Xva, yva, lr=3e-4, epochs=20, bs=128):
    model = model.to(device)
    cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=ytr)
    cw = torch.tensor(cw, dtype=torch.float32).to(device)
    loss_fn = FocalLoss(gamma=2.0, weight=cw, label_smoothing=0.02)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    tr_dl = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.long)),
        batch_size=bs, shuffle=True
    )
    va_dl = DataLoader(
        TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                      torch.tensor(yva, dtype=torch.long)),
        batch_size=bs, shuffle=False
    )

    best = -1.0
    best_state = None
    patience = 6
    bad = 0

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in va_dl:
                xb = xb.to(device)
                preds.append(torch.argmax(model(xb), dim=1).cpu().numpy())
        preds = np.concatenate(preds)

        va_f1 = f1_score(yva, preds, average="binary", zero_division=0)
        va_acc = accuracy_score(yva, preds)
        sched.step(va_f1)
        print(f"Epoch {ep:02d} val_acc={va_acc:.4f} val_f1={va_f1:.4f}")

        if va_f1 > best:
            best = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

print("Training text model...")
text_model = train_classifier(TextMLP(), X_text_tr, y_train, X_text_va, y_val, lr=2e-4, epochs=20, bs=128)

print("Training image model...")
image_model = train_classifier(ImageMLP(), X_img_tr, y_train, X_img_va, y_val, lr=5e-4, epochs=20, bs=128)

# ============================================================
# Logits + probs (for stacking + MCDM)
# ============================================================
@torch.no_grad()
def get_logits(model, X, bs=256):
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=bs, shuffle=False)
    model.eval()
    out = []
    for (xb,) in dl:
        xb = xb.to(device)
        out.append(model(xb).cpu().numpy())
    return np.vstack(out)

def softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-9)

text_logits_va = get_logits(text_model, X_text_va)
img_logits_va  = get_logits(image_model, X_img_va)
text_logits_te = get_logits(text_model, X_text_te)
img_logits_te  = get_logits(image_model, X_img_te)

text_probs_va = softmax_np(text_logits_va)
img_probs_va  = softmax_np(img_logits_va)
text_probs_te = softmax_np(text_logits_te)
img_probs_te  = softmax_np(img_logits_te)

print("\nVAL text-only acc:", round(accuracy_score(y_val, np.argmax(text_probs_va,1))*100, 2), "%")
print("VAL img-only acc :", round(accuracy_score(y_val, np.argmax(img_probs_va,1))*100, 2), "%")

# ============================================================
# STACKING fusion (train on VAL, test on TEST)
# ============================================================
def entropy(P):
    P = np.clip(P, 1e-9, 1.0)
    return -(P * np.log(P)).sum(axis=1, keepdims=True)

def margin(P):
    s = np.sort(P, axis=1)
    return (s[:, -1] - s[:, -2]).reshape(-1, 1)

X_meta_va = np.hstack([
    text_logits_va, img_logits_va,
    entropy(text_probs_va), entropy(img_probs_va),
    margin(text_probs_va), margin(img_probs_va),
]).astype(np.float32)

X_meta_te = np.hstack([
    text_logits_te, img_logits_te,
    entropy(text_probs_te), entropy(img_probs_te),
    margin(text_probs_te), margin(img_probs_te),
]).astype(np.float32)

mu = X_meta_va.mean(axis=0, keepdims=True)
sd = X_meta_va.std(axis=0, keepdims=True) + 1e-6
X_meta_va = (X_meta_va - mu) / sd
X_meta_te = (X_meta_te - mu) / sd

meta = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",   # supported in sklearn LogisticRegression docs
    C=2.0
)
meta.fit(X_meta_va, y_val)

stack_pred_te = meta.predict(X_meta_te)
print("\nSTACKING TEST acc:", round(accuracy_score(y_test, stack_pred_te)*100, 2), "%")
print("STACKING TEST F1 :", round(f1_score(y_test, stack_pred_te, average="binary", zero_division=0)*100, 2), "%")

# ============================================================
# MCDM (alternatives = 2 classes, criteria = 2 modalities)
# D = (2,2): rows=classes, col0=text prob, col1=image prob
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

METHODS = {
    "TOPSIS": lambda D, W, theta: topsis(D, W),
    "RAFSI":  lambda D, W, theta: rafsi(D, W),
    "TODIM":  lambda D, W, theta: todim(D, W, theta=theta),
    "MARCOS": lambda D, W, theta: marcos(D, W),
    "EDAS":   lambda D, W, theta: edas(D, W),
}

def predict_mcdm(method_name, text_probs, img_probs, W, theta=1.0):
    fn = METHODS[method_name]
    pred = []
    for i in range(len(text_probs)):
        D = np.stack([text_probs[i], img_probs[i]], axis=1)  # (2,2)
        score = fn(D, W, theta)
        pred.append(int(np.argmax(score)))
    return np.array(pred, dtype=np.int64)

def temp_scale_probs(P, T):
    P = np.clip(P, 1e-9, 1.0)
    logP = np.log(P) / float(T)
    expv = np.exp(logP - logP.max(axis=1, keepdims=True))
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-9)

weight_candidates = [np.array([w, 1.0-w], dtype=np.float32) for w in
                     [0.98,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]]
theta_candidates = [0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0]
temp_candidates  = [0.8, 1.0, 1.2, 1.5, 2.0]

val_results = []
for m in METHODS.keys():
    for W in weight_candidates:
        thetas = theta_candidates if m == "TODIM" else [1.0]
        for theta in thetas:
            for T in temp_candidates:
                tp = temp_scale_probs(text_probs_va, T)
                ip = temp_scale_probs(img_probs_va, T)
                pred = predict_mcdm(m, tp, ip, W=W, theta=theta)
                acc = accuracy_score(y_val, pred)
                f1  = f1_score(y_val, pred, average="binary", zero_division=0)
                val_results.append((acc, f1, m, float(W[0]), float(W[1]), float(theta), float(T)))

# SOFT fusion baseline
for tw in [0.98,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]:
    for T in temp_candidates:
        tp = temp_scale_probs(text_probs_va, T)
        ip = temp_scale_probs(img_probs_va, T)
        pred = np.argmax(tw*tp + (1.0-tw)*ip, axis=1)
        acc = accuracy_score(y_val, pred)
        f1  = f1_score(y_val, pred, average="binary", zero_division=0)
        val_results.append((acc, f1, "SOFT", float(tw), float(1.0-tw), 1.0, float(T)))

val_results.sort(key=lambda x: x[0], reverse=True)

print("\nTop 10 configs on VAL (by Accuracy):")
for i, (acc, f1, m, w0, w1, theta, T) in enumerate(val_results[:10], 1):
    extra = f"theta={theta:.2f}" if m == "TODIM" else ""
    print(f"{i:02d}. {m:6s} val_acc={acc*100:.2f}% val_f1={f1*100:.2f}% W=[{w0:.2f},{w1:.2f}] T={T:.2f} {extra}")

best = val_results[0]
_, _, best_m, best_w0, best_w1, best_theta, best_T = best
best_W = np.array([best_w0, best_w1], dtype=np.float32)

tp_te = temp_scale_probs(text_probs_te, best_T)
ip_te = temp_scale_probs(img_probs_te, best_T)

if best_m == "SOFT":
    test_pred = np.argmax(best_w0*tp_te + best_w1*ip_te, axis=1)
else:
    test_pred = predict_mcdm(best_m, tp_te, ip_te, W=best_W, theta=best_theta)

print("\nBest MCDM config on VAL:", best_m, "W=", best_W, "theta=", best_theta, "T=", best_T)
print("MCDM TEST accuracy:", round(accuracy_score(y_test, test_pred)*100, 2), "%")
print("MCDM TEST F1:", round(f1_score(y_test, test_pred, average="binary", zero_division=0)*100, 2), "%")
