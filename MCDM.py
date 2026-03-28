import os
import zipfile
import pickle
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ============================================================
# Setup
# ============================================================
SEED = 42
BASE_SEEDS = [11, 21, 31, 41, 51]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

LABELMAP = {"negative": 0, "positive": 1}
INV_LABELMAP = {0: "negative", 1: "positive"}

ZIPPATH = Path("/home/neha/Dataset/mosi.zip")
EXTRACTROOT = Path("/home/neha/Dataset/mosi_extracted")
EXTRACTROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 50
LR = 8e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
LABEL_SMOOTH = 0.05
HIDDEN = 256
PROJ = 128
DROPOUT = 0.35

# ============================================================
# Load PKLs
# ============================================================
def maybe_extract_zip(zip_path: Path, extract_root: Path):
    probe = list(extract_root.rglob("audio_dict.pkl"))
    if probe:
        print("PKL files already present, skipping unzip.")
        return
    print("Extracting:", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

def load_pkl(name: str):
    matches = list(EXTRACTROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {EXTRACTROOT}")
    path = matches[0]
    print("Loading", path)
    with open(path, "rb") as f:
        return pickle.load(f)

maybe_extract_zip(ZIPPATH, EXTRACTROOT)
audiodict = load_pkl("audio_dict.pkl")
visualdict = load_pkl("processed_visual_dict.pkl")
textembdict = load_pkl("text_emb.pkl")
labeldict = load_pkl("label_dict.pkl")

print("Loaded:", len(textembdict), len(visualdict), len(audiodict), len(labeldict))

# ============================================================
# Helpers
# ============================================================
def encode_label(raw):
    s = str(raw).strip().lower()
    if s == "neutral":
        raise ValueError("neutral dropped")
    if s not in LABELMAP:
        raise ValueError(f"Unknown label: {raw}")
    return LABELMAP[s]

def get_feat(uid, dct):
    x = np.asarray(dct[uid], dtype=np.float32)
    if x.ndim == 2:
        return x.mean(axis=0).astype(np.float32)
    if x.ndim == 1:
        return x.astype(np.float32)
    return None

def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{name}")
    print("Acc:", round(acc * 100, 2))
    print("Prec:", round(prec * 100, 2))
    print("Rec:", round(rec * 100, 2))
    print("F1:", round(f1 * 100, 2))
    print("CM:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    return acc, prec, rec, f1

# ============================================================
# Split IDs
# ============================================================
all_ids = []
for uid in textembdict.keys():
    if uid in audiodict and uid in visualdict and uid in labeldict:
        lab = str(labeldict[uid]).strip().lower()
        if lab in LABELMAP:
            all_ids.append(uid)

all_ids = np.array(sorted(all_ids))
rng = np.random.RandomState(SEED)
rng.shuffle(all_ids)

n = len(all_ids)
n_train = int(0.7 * n)
n_val = int(0.1 * n)

train_ids = all_ids[:n_train]
val_ids = all_ids[n_train:n_train + n_val]
test_ids = all_ids[n_train + n_val:]

print("Binary IDs:", n, "| Train:", len(train_ids), "| Val:", len(val_ids), "| Test:", len(test_ids))

# ============================================================
# Feature matrices
# ============================================================
def build_split_features(id_list):
    Xt, Xv, Xa, y = [], [], [], []
    for uid in id_list:
        try:
            yy = encode_label(labeldict[uid])
        except Exception:
            continue
        t = get_feat(uid, textembdict)
        v = get_feat(uid, visualdict)
        a = get_feat(uid, audiodict)
        if t is None or v is None or a is None:
            continue
        Xt.append(t)
        Xv.append(v)
        Xa.append(a)
        y.append(yy)
    return np.asarray(Xt), np.asarray(Xv), np.asarray(Xa), np.asarray(y, dtype=np.int64)

Xtext_tr, Xvis_tr, Xaud_tr, y_tr = build_split_features(train_ids)
Xtext_va, Xvis_va, Xaud_va, y_va = build_split_features(val_ids)
Xtext_te, Xvis_te, Xaud_te, y_te = build_split_features(test_ids)

print("Shapes:")
print("Text:", Xtext_tr.shape)
print("Vis :", Xvis_tr.shape)
print("Aud :", Xaud_tr.shape)

# ============================================================
# Standardize
# ============================================================
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(axis=0, keepdims=True)
        self.sd = X.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mu) / self.sd

sc_t = Standardizer().fit(Xtext_tr)
sc_v = Standardizer().fit(Xvis_tr)
sc_a = Standardizer().fit(Xaud_tr)

Xtext_tr_s = sc_t.transform(Xtext_tr)
Xtext_va_s = sc_t.transform(Xtext_va)
Xtext_te_s = sc_t.transform(Xtext_te)

Xvis_tr_s = sc_v.transform(Xvis_tr)
Xvis_va_s = sc_v.transform(Xvis_va)
Xvis_te_s = sc_v.transform(Xvis_te)

Xaud_tr_s = sc_a.transform(Xaud_tr)
Xaud_va_s = sc_a.transform(Xaud_va)
Xaud_te_s = sc_a.transform(Xaud_te)

# ============================================================
# Unimodal classifiers
# ============================================================
text_clf = LogisticRegression(max_iter=5000, class_weight="balanced")
vis_clf = LogisticRegression(max_iter=5000, class_weight="balanced")
aud_clf = LogisticRegression(max_iter=5000, class_weight="balanced")

text_clf.fit(Xtext_tr_s, y_tr)
vis_clf.fit(Xvis_tr_s, y_tr)
aud_clf.fit(Xaud_tr_s, y_tr)

tp_va = text_clf.predict_proba(Xtext_va_s)
vp_va = vis_clf.predict_proba(Xvis_va_s)
ap_va = aud_clf.predict_proba(Xaud_va_s)

tp_te = text_clf.predict_proba(Xtext_te_s)
vp_te = vis_clf.predict_proba(Xvis_te_s)
ap_te = aud_clf.predict_proba(Xaud_te_s)

print_metrics("Text VAL", y_va, np.argmax(tp_va, 1))
print_metrics("Visual VAL", y_va, np.argmax(vp_va, 1))
print_metrics("Audio VAL", y_va, np.argmax(ap_va, 1))

# ============================================================
# Datasets
# ============================================================
class FusionDS(Dataset):
    def __init__(self, Xt, Xv, Xa, y):
        self.Xt = torch.tensor(Xt, dtype=torch.float32)
        self.Xv = torch.tensor(Xv, dtype=torch.float32)
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.Xt[idx], self.Xv[idx], self.Xa[idx], self.y[idx]

# ============================================================
# Model: Cross-attention fusion
# ============================================================
class CrossAttnFusionNet(nn.Module):
    def __init__(self, dt, dv, da, proj=128, hidden=256, dropout=0.35, heads=4):
        super().__init__()
        self.tproj = nn.Sequential(nn.Linear(dt, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(dropout))
        self.vproj = nn.Sequential(nn.Linear(dv, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(dropout))
        self.aproj = nn.Sequential(nn.Linear(da, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(dropout))

        self.tv = nn.MultiheadAttention(proj, heads, dropout=dropout, batch_first=True)
        self.ta = nn.MultiheadAttention(proj, heads, dropout=dropout, batch_first=True)
        self.vt = nn.MultiheadAttention(proj, heads, dropout=dropout, batch_first=True)
        self.va = nn.MultiheadAttention(proj, heads, dropout=dropout, batch_first=True)
        self.at = nn.MultiheadAttention(proj, heads, dropout=dropout, batch_first=True)
        self.av = nn.MultiheadAttention(proj, heads, dropout=dropout, batch_first=True)

        self.gate = nn.Sequential(nn.Linear(proj * 6, 6), nn.Softmax(dim=-1))

        self.fuse = nn.Sequential(
            nn.Linear(proj * 6, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, 2)

    def forward(self, xt, xv, xa, return_hidden=False):
        t = self.tproj(xt).unsqueeze(1)
        v = self.vproj(xv).unsqueeze(1)
        a = self.aproj(xa).unsqueeze(1)

        t_v, _ = self.tv(t, v, v)
        t_a, _ = self.ta(t, a, a)
        v_t, _ = self.vt(v, t, t)
        v_a, _ = self.va(v, a, a)
        a_t, _ = self.at(a, t, t)
        a_v, _ = self.av(a, v, v)

        feats = torch.cat([
            t_v.squeeze(1),
            t_a.squeeze(1),
            v_t.squeeze(1),
            v_a.squeeze(1),
            a_t.squeeze(1),
            a_v.squeeze(1),
        ], dim=-1)

        g = self.gate(feats)
        gated = torch.cat([
            t_v.squeeze(1) * g[:, 0:1],
            t_a.squeeze(1) * g[:, 1:2],
            v_t.squeeze(1) * g[:, 2:3],
            v_a.squeeze(1) * g[:, 3:4],
            a_t.squeeze(1) * g[:, 4:5],
            a_v.squeeze(1) * g[:, 5:6],
        ], dim=-1)

        hidden = self.fuse(gated)
        logits = self.out(hidden)
        if return_hidden:
            return logits, hidden, g
        return logits

def label_smoothing_loss(logits, target, eps=0.05):
    n = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(eps / (n - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def train_one_model(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CrossAttnFusionNet(Xtext_tr_s.shape[1], Xvis_tr_s.shape[1], Xaud_tr_s.shape[1]).to(device)

    tr_dl = DataLoader(FusionDS(Xtext_tr_s, Xvis_tr_s, Xaud_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(FusionDS(Xtext_va_s, Xvis_va_s, Xaud_va_s, y_va), batch_size=BATCH_SIZE, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_acc, best_state, bad = -1, None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xt, xv, xa, yb in tr_dl:
            xt, xv, xa, yb = xt.to(device), xv.to(device), xa.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xt, xv, xa)
            loss = label_smoothing_loss(logits, yb, eps=LABEL_SMOOTH)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xt, xv, xa, _ in va_dl:
                xt, xv, xa = xt.to(device), xv.to(device), xa.to(device)
                preds.append(torch.argmax(model(xt, xv, xa), dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        acc = accuracy_score(y_va, preds)

        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc

@torch.no_grad()
def infer_outputs(model, Xt, Xv, Xa):
    model.eval()
    ds = FusionDS(Xt, Xv, Xa, np.zeros(len(Xt), dtype=np.int64))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    probs, logits_list, hidden_list, gate_list = [], [], [], []
    for xt, xv, xa, _ in dl:
        xt, xv, xa = xt.to(device), xv.to(device), xa.to(device)
        logits, hidden, gate = model(xt, xv, xa, return_hidden=True)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        logits_list.append(logits.cpu().numpy())
        hidden_list.append(hidden.cpu().numpy())
        gate_list.append(gate.cpu().numpy())
    return (
        np.vstack(probs),
        np.vstack(logits_list),
        np.vstack(hidden_list),
        np.vstack(gate_list),
    )

# ============================================================
# Ensemble training
# ============================================================
val_probs_all, val_logits_all, val_hidden_all, val_gate_all = [], [], [], []
test_probs_all, test_logits_all, test_hidden_all, test_gate_all = [], [], [], []

for s in BASE_SEEDS:
    model, best_acc = train_one_model(s)
    pv, lv, hv, gv = infer_outputs(model, Xtext_va_s, Xvis_va_s, Xaud_va_s)
    pt, lt, ht, gt = infer_outputs(model, Xtext_te_s, Xvis_te_s, Xaud_te_s)

    val_probs_all.append(pv)
    val_logits_all.append(lv)
    val_hidden_all.append(hv)
    val_gate_all.append(gv)

    test_probs_all.append(pt)
    test_logits_all.append(lt)
    test_hidden_all.append(ht)
    test_gate_all.append(gt)

    print(f"Seed {s} best val acc: {best_acc*100:.2f}%")

fusion_va = np.mean(np.stack(val_probs_all, axis=0), axis=0)
fusion_te = np.mean(np.stack(test_probs_all, axis=0), axis=0)

logits_va = np.mean(np.stack(val_logits_all, axis=0), axis=0)
logits_te = np.mean(np.stack(test_logits_all, axis=0), axis=0)

hidden_va = np.mean(np.stack(val_hidden_all, axis=0), axis=0)
hidden_te = np.mean(np.stack(test_hidden_all, axis=0), axis=0)

gate_va = np.mean(np.stack(val_gate_all, axis=0), axis=0)
gate_te = np.mean(np.stack(test_gate_all, axis=0), axis=0)

print_metrics("ENSEMBLE FUSION VAL", y_va, np.argmax(fusion_va, axis=1))
print_metrics("ENSEMBLE FUSION TEST", y_te, np.argmax(fusion_te, axis=1))

# ============================================================
# MCDM on hidden/logit features
# ============================================================
def entropy_from_probvec(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

def margin_from_probvec(p):
    p = np.asarray(p, dtype=np.float32)
    s = np.sort(p)
    return float(s[-1] - s[-2])

def ensemble_agreement(p, ep):
    return 1.0 if int(np.argmax(p)) == int(np.argmax(ep)) else 0.0

def reliability_score(p, ep):
    c1 = float(np.max(p))
    c2 = float(np.max(ep))
    agree = ensemble_agreement(p, ep)
    gap = abs(c1 - c2)
    return 0.5 * agree + 0.5 * (1.0 - min(gap, 1.0))

def build_2x9_matrix(prob, logits, hidden, gate, ensemble_prob):
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
    gate = np.asarray(gate, dtype=np.float32).reshape(-1)
    ep = np.asarray(ensemble_prob, dtype=np.float32).reshape(-1)

    p = float(prob[1])
    c = float(np.max(prob))
    e = entropy_from_probvec(prob)
    m = margin_from_probvec(prob)
    agr = ensemble_agreement(prob, ep)
    rel = reliability_score(prob, ep)
    lnorm = float(np.linalg.norm(logits))
    hnorm = float(np.linalg.norm(hidden))
    gmean = float(np.mean(gate))

    row_neg = np.array([
        1.0 - p, 1.0 - c, e,
        1.0 - m, 1.0 - agr, 1.0 - rel,
        1.0 - lnorm / (1.0 + lnorm), 1.0 - hnorm / (1.0 + hnorm), 1.0 - gmean
    ], dtype=np.float32)

    row_pos = np.array([
        p, c, 1.0 - e,
        m, agr, rel,
        lnorm / (1.0 + lnorm), hnorm / (1.0 + hnorm), gmean
    ], dtype=np.float32)

    return np.vstack([row_neg, row_pos])

def minmax_norm(D):
    mn = D.min(axis=0)
    mx = D.max(axis=0)
    denom = np.where((mx - mn) == 0, 1.0, (mx - mn))
    return (D - mn) / denom

def topsis(D, W):
    R = D / (np.sqrt((D ** 2).sum(axis=0)) + 1e-9)
    V = R * W
    ideal = V.max(axis=0)
    nadir = V.min(axis=0)
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
    "RAFSI": lambda D, W, theta: rafsi(D, W),
    "TODIM": lambda D, W, theta: todim(D, W, theta=theta),
    "MARCOS": lambda D, W, theta: marcos(D, W),
    "EDAS": lambda D, W, theta: edas(D, W),
}

def weights_9_from_3(W3):
    wt, wv, wa = np.asarray(W3, dtype=np.float32)
    W9 = np.array([
        wt * 0.25, wt * 0.15, wt * 0.10,
        wv * 0.25, wv * 0.15, wv * 0.10,
        wa * 0.25, wa * 0.15, wa * 0.10,
    ], dtype=np.float32)
    return W9 / (W9.sum() + 1e-9)

def predict_mcdm(method_name, prob, logits, hidden, gate, ensemble_prob, W3, theta=1.0):
    fn = MCDM_METHODS[method_name]
    W = weights_9_from_3(W3)
    preds = []
    for i in range(len(prob)):
        D = build_2x9_matrix(prob[i], logits[i], hidden[i], gate[i], ensemble_prob[i])
        score = fn(D, W, theta)
        preds.append(int(np.argmax(score)))
    return np.array(preds, dtype=np.int64)

# ============================================================
# Tune and report
# ============================================================
WEIGHT_CANDIDATES = [
    np.array([0.55, 0.25, 0.20], dtype=np.float32),
    np.array([0.60, 0.20, 0.20], dtype=np.float32),
    np.array([0.65, 0.20, 0.15], dtype=np.float32),
    np.array([0.70, 0.15, 0.15], dtype=np.float32),
    np.array([0.75, 0.10, 0.15], dtype=np.float32),
]

val_results = []
for m in MCDM_METHODS:
    for W in WEIGHT_CANDIDATES:
        thetas = [1.0] if m != "TODIM" else [0.5, 0.75, 1.0, 1.25]
        for theta in thetas:
            pred = predict_mcdm(m, fusion_va, logits_va, hidden_va, gate_va, fusion_va, W3=W, theta=theta)
            acc = accuracy_score(y_va, pred)
            f1 = f1_score(y_va, pred, zero_division=0)
            val_results.append((acc, f1, m, W.copy(), float(theta)))

val_results.sort(key=lambda x: x[0], reverse=True)

print("\nTop 10 configs on VAL:")
for i, r in enumerate(val_results[:10], 1):
    acc, f1, m, W, theta = r
    print(f"{i:02d}. {m:6s} val_acc={acc*100:.2f}% val_f1={f1*100:.2f}% W={np.round(W,2)} theta={theta:.2f}")

print("\nBest TEST performance per fusion method:")
for m in ["TOPSIS", "RAFSI", "TODIM", "MARCOS", "EDAS"]:
    cand = [r for r in val_results if r[2] == m]
    if not cand:
        continue
    acc, f1, _, W, theta = sorted(cand, key=lambda x: x[0], reverse=True)[0]
    test_pred = predict_mcdm(m, fusion_te, logits_te, hidden_te, gate_te, fusion_te, W3=W, theta=theta)
    print_metrics(f"{m} TEST", y_te, test_pred)

print_metrics("ENSEMBLE FUSION TEST (final)", y_te, np.argmax(fusion_te, axis=1))
