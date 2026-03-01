# ============================================================
# MOSI Binary (drop neutral) + Experts + MCDM + Meta Stacking
# Target: maximize TEST accuracy without leakage (train->val->test)
# ============================================================

import zipfile
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# -------------------------
# CELL 1: Seed + device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------
# CELL 2: Load PKLs
# -------------------------
ZIPPATH = Path("/home/neha/Dataset/mosi.zip")
EXTRACTROOT = Path("/home/neha/Dataset/mosi_extracted")
EXTRACTROOT.mkdir(parents=True, exist_ok=True)

def maybe_extract_zip(zip_path: Path, extract_root: Path):
    probe = list(extract_root.rglob("audiodict.pkl"))
    if len(probe) > 0:
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


# -------------------------
# CELL 3: Binary labels (drop neutral)
# -------------------------
LABELS = ["negative", "positive"]
LABELMAP = {"negative": 0, "positive": 1}
INVLABELMAP = {v: k for k, v in LABELMAP.items()}

def encode_label(raw):
    s = str(raw).strip().lower()
    if s == "neutral":
        raise ValueError("neutral dropped")
    if s not in LABELMAP:
        raise ValueError(f"Unknown label {raw}")
    return LABELMAP[s]


# -------------------------
# CELL 4: Build IDs and split (drop neutrals BEFORE split)
# -------------------------
all_ids = []
for uid in textembdict.keys():
    if uid not in audiodict or uid not in visualdict or uid not in labeldict:
        continue
    lab = str(labeldict[uid]).strip().lower()
    if lab == "neutral":
        continue
    if lab not in ("negative", "positive"):
        continue
    all_ids.append(uid)

all_ids = np.array(sorted(all_ids))
rng = np.random.RandomState(SEED)
rng.shuffle(all_ids)

n = len(all_ids)
n_train = int(0.7 * n)
n_val = int(0.1 * n)

train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train + n_val]
test_ids  = all_ids[n_train + n_val:]

print("Binary IDs:", n, "| Train:", len(train_ids), "| Val:", len(val_ids), "| Test:", len(test_ids))


# -------------------------
# CELL 5: Feature building (same pooling as your script)
# -------------------------
def build_split_features(id_list):
    Xtext, Ximg, Xaud, y = [], [], [], []
    for uid in id_list:
        try:
            yy = encode_label(labeldict[uid])
        except Exception:
            continue

        a = np.asarray(audiodict[uid], dtype=np.float32)
        v = np.asarray(visualdict[uid], dtype=np.float32)
        t = np.asarray(textembdict[uid], dtype=np.float32)

        if a.ndim != 2 or v.ndim != 2:
            continue
        if t.ndim == 2:
            tv = t[0].astype(np.float32)   # keep your current behavior
        elif t.ndim == 1:
            tv = t.astype(np.float32)
        else:
            continue

        iv = v.mean(axis=0).astype(np.float32)
        av = a.mean(axis=0).astype(np.float32)

        Xtext.append(tv); Ximg.append(iv); Xaud.append(av); y.append(yy)

    if len(y) == 0:
        return None

    return (np.asarray(Xtext, np.float32),
            np.asarray(Ximg, np.float32),
            np.asarray(Xaud, np.float32),
            np.asarray(y, np.int64))

train_pack = build_split_features(train_ids)
val_pack   = build_split_features(val_ids)
test_pack  = build_split_features(test_ids)

if train_pack is None or val_pack is None or test_pack is None:
    raise RuntimeError("Empty split after filtering.")

Xtext_tr, Ximg_tr, Xaud_tr, y_tr = train_pack
Xtext_va, Ximg_va, Xaud_va, y_va = val_pack
Xtext_te, Ximg_te, Xaud_te, y_te = test_pack

print("Shapes:", Xtext_tr.shape, Ximg_tr.shape, Xaud_tr.shape)


# -------------------------
# CELL 6: Standardize per modality (fit on train only)
# -------------------------
class Standardizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X):
        self.mu = X.mean(axis=0, keepdims=True)
        self.sd = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd

sc_text = Standardizer().fit(Xtext_tr)
sc_img  = Standardizer().fit(Ximg_tr)
sc_aud  = Standardizer().fit(Xaud_tr)

Xtext_tr_s = sc_text.transform(Xtext_tr)
Xtext_va_s = sc_text.transform(Xtext_va)
Xtext_te_s = sc_text.transform(Xtext_te)

Ximg_tr_s = sc_img.transform(Ximg_tr)
Ximg_va_s = sc_img.transform(Ximg_va)
Ximg_te_s = sc_img.transform(Ximg_te)

Xaud_tr_s = sc_aud.transform(Xaud_tr)
Xaud_va_s = sc_aud.transform(Xaud_va)
Xaud_te_s = sc_aud.transform(Xaud_te)


# -------------------------
# CELL 7: Loss (Focal) + utilities
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        return fl.mean()

def metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    print(f"\n{name}")
    print("Acc:", round(acc*100, 2))
    print("Prec:", round(prec*100, 2))
    print("Rec:", round(rec*100, 2))
    print("F1:", round(f1*100, 2))
    print("CM:\n", confusion_matrix(y_true, y_pred, labels=[0,1]))
    return acc


# -------------------------
# CELL 8: Expert model
# -------------------------
class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden=(512,256,128), dropout=0.3, num_classes=2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x):
        return self.out(self.backbone(x))


# -------------------------
# CELL 9: Train expert with val selection (NO leakage)
# -------------------------
def train_expert(Xtr, ytr, Xva, yva, hidden, dropout, lr, epochs, batchsize, use_focal=True):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=ytr)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=device)

    model = ExpertMLP(Xtr.shape[1], hidden=hidden, dropout=dropout, num_classes=2).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    crit = FocalLoss(alpha=cls_w_t, gamma=2.0) if use_focal else nn.CrossEntropyLoss(weight=cls_w_t)

    best_acc = -1.0
    best_state = None
    patience = 10
    bad = 0
    N = len(ytr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for st in range(0, N, batchsize):
            idx = perm[st:st+batchsize]
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_t[idx])
            loss = crit(out, ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t).argmax(dim=1).detach().cpu().numpy()
        acc_va = accuracy_score(yva, pred_va)

        if acc_va > best_acc + 1e-5:
            best_acc = acc_va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_acc


def train_expert_best(Xtr, ytr, Xva, yva, max_trials=20, seed=42):
    rng = np.random.RandomState(seed)
    grid = []
    for hidden in [(512,256,128), (1024,512,256), (512,256), (768,256)]:
        for dropout in [0.1, 0.2, 0.3, 0.4]:
            for lr in [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]:
                for epochs in [60, 90, 120]:
                    for bs in [32, 64, 128]:
                        grid.append((hidden, dropout, lr, epochs, bs))
    rng.shuffle(grid)
    grid = grid[:max_trials]

    best = None
    best_acc = -1.0
    for (hidden, dropout, lr, epochs, bs) in grid:
        m, acc = train_expert(Xtr, ytr, Xva, yva,
                              hidden=hidden, dropout=dropout, lr=lr,
                              epochs=epochs, batchsize=bs, use_focal=True)
        if acc > best_acc:
            best_acc = acc
            best = (m, (hidden, dropout, lr, epochs, bs))
        print("trial", (hidden, dropout, lr, epochs, bs), "val_acc", round(acc,4))

    return best[0], best[1], best_acc


# -------------------------
# CELL 10: Train 3 experts
# -------------------------
text_model, text_params, text_val = train_expert_best(Xtext_tr_s, y_tr, Xtext_va_s, y_va, max_trials=18, seed=SEED)
img_model,  img_params,  img_val  = train_expert_best(Ximg_tr_s,  y_tr, Ximg_va_s,  y_va, max_trials=18, seed=SEED+1)
aud_model,  aud_params,  aud_val  = train_expert_best(Xaud_tr_s,  y_tr, Xaud_va_s,  y_va, max_trials=18, seed=SEED+2)

print("\nBest val accs:", "text", text_val, "img", img_val, "aud", aud_val)


# -------------------------
# CELL 11: Get logits/probs
# -------------------------
@torch.no_grad()
def logits_probs(model, X):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    lg = model(Xt).detach().cpu().numpy()
    pr = np.exp(lg - lg.max(axis=1, keepdims=True))
    pr = pr / (pr.sum(axis=1, keepdims=True) + 1e-9)
    return lg, pr

lt_te, pT_te = logits_probs(text_model, Xtext_te_s)
li_te, pI_te = logits_probs(img_model,  Ximg_te_s)
la_te, pA_te = logits_probs(aud_model,  Xaud_te_s)

# expert accuracies
metrics(y_te, lt_te.argmax(axis=1), "Text expert")
metrics(y_te, li_te.argmax(axis=1), "Image expert")
metrics(y_te, la_te.argmax(axis=1), "Audio expert")

# avg-logits fusion
avg_logits = (lt_te + li_te + la_te) / 3.0
pred_avg = avg_logits.argmax(axis=1)
metrics(y_te, pred_avg, "Avg-logits fusion (experts)")


# -------------------------
# CELL 12: MCDM for binary (2 alternatives x 3 criteria)
# -------------------------
METHODS = ["SAW", "TOPSIS", "RAFSI", "TODIM", "MARCOS"]

def minmax_norm_cols(D):
    mn = D.min(axis=0, keepdims=True)
    mx = D.max(axis=0, keepdims=True)
    diff = np.where((mx - mn) == 0, 1.0, (mx - mn))
    return (D - mn) / diff

def decision_matrix_from_probs(p_text, p_img, p_aud):
    return np.stack([p_text, p_img, p_aud], axis=1)  # (2,3)

def mcdm_saw(D, W):
    R = minmax_norm_cols(D)
    return (R * W).sum(axis=1)

def mcdm_topsis(D, W):
    denom = np.sqrt((D ** 2).sum(axis=0, keepdims=True)) + 1e-9
    R = D / denom
    V = R * W
    ideal_best = V.max(axis=0, keepdims=True)
    ideal_worst = V.min(axis=0, keepdims=True)
    dpos = np.linalg.norm(V - ideal_best, axis=1)
    dneg = np.linalg.norm(V - ideal_worst, axis=1)
    return dneg / (dpos + dneg + 1e-9)

def mcdm_rafsi(D, W):
    R = minmax_norm_cols(D)
    ranks = R.argsort(axis=0).argsort(axis=0).astype(np.float32)
    return (ranks * W).sum(axis=1)

def mcdm_todim(D, W, theta=1.0):
    R = minmax_norm_cols(D)
    m = R.shape[0]  # 2
    score = np.zeros((m,), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            diff = R[i] - R[j]
            score[i] += (W * np.maximum(diff, 0)).sum() - theta * (W * np.maximum(-diff, 0)).sum()
    return score

def mcdm_marcos(D, W):
    R = minmax_norm_cols(D)
    S = (R * W).sum(axis=1)
    return S / (S.sum() + 1e-9)

def mcdm_scores_for_sample(p_text, p_img, p_aud, method, W=None):
    if W is None:
        W = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    D = decision_matrix_from_probs(p_text, p_img, p_aud)
    if method == "SAW":
        return mcdm_saw(D, W)
    if method == "TOPSIS":
        return mcdm_topsis(D, W)
    if method == "RAFSI":
        return mcdm_rafsi(D, W)
    if method == "TODIM":
        return mcdm_todim(D, W, theta=1.0)
    if method == "MARCOS":
        return mcdm_marcos(D, W)
    raise ValueError("Unknown method")

def build_mcdm_features(pT, pI, pA, method):
    N = pT.shape[0]
    X = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        X[i] = mcdm_scores_for_sample(pT[i], pI[i], pA[i], method=method)
    return X

mcdm_preds = {}
for m in METHODS:
    Xm = build_mcdm_features(pT_te, pI_te, pA_te, m)
    pred = Xm.argmax(axis=1)
    mcdm_preds[m] = pred
    metrics(y_te, pred, f"MCDM-{m} (direct)")


# -------------------------
# CELL 13: Meta-stacker over ALL signals (best final fusion)
# Features:
#   - expert probs: 2+2+2 = 6
#   - MCDM scores (each method gives 2): 5*2=10
# Total = 16 dims
# -------------------------
class MetaNN(nn.Module):
    def __init__(self, in_dim, hidden=(64,32), dropout=0.2, num_classes=2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_meta_X(pT, pI, pA):
    feats = [pT, pI, pA]  # each (N,2)
    for m in METHODS:
        feats.append(build_mcdm_features(pT, pI, pA, m))  # (N,2)
    return np.concatenate(feats, axis=1).astype(np.float32)  # (N,16)

# Build meta train/val/test using expert models (needs probs for each split)
lt_tr, pT_tr = logits_probs(text_model, Xtext_tr_s)
li_tr, pI_tr = logits_probs(img_model,  Ximg_tr_s)
la_tr, pA_tr = logits_probs(aud_model,  Xaud_tr_s)

lt_va, pT_va = logits_probs(text_model, Xtext_va_s)
li_va, pI_va = logits_probs(img_model,  Ximg_va_s)
la_va, pA_va = logits_probs(aud_model,  Xaud_va_s)

Xmeta_tr = build_meta_X(pT_tr, pI_tr, pA_tr)
Xmeta_va = build_meta_X(pT_va, pI_va, pA_va)
Xmeta_te = build_meta_X(pT_te, pI_te, pA_te)

# train meta with early stopping on val
def train_meta(Xtr, ytr, Xva, yva, epochs=200, lr=1e-3, dropout=0.2, batchsize=64):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=ytr)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw_t)

    model = MetaNN(in_dim=Xtr.shape[1], hidden=(128,64), dropout=dropout, num_classes=2).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = -1.0
    best_state = None
    patience = 15
    bad = 0
    N = len(ytr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for st in range(0, N, batchsize):
            idx = perm[st:st+batchsize]
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_t[idx])
            loss = crit(out, ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t).argmax(dim=1).detach().cpu().numpy()
        acc_va = accuracy_score(yva, pred_va)

        if acc_va > best_acc + 1e-5:
            best_acc = acc_va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc

meta_model, meta_val = train_meta(Xmeta_tr, y_tr, Xmeta_va, y_va, epochs=220, lr=8e-4, dropout=0.2, batchsize=64)
print("\nMeta best val acc:", meta_val)

meta_model.eval()
with torch.no_grad():
    pred_meta = meta_model(torch.tensor(Xmeta_te, dtype=torch.float32, device=device)).argmax(dim=1).detach().cpu().numpy()

metrics(y_te, pred_meta, "ALL_FUSION (Experts + MCDM + MetaNN)")

# save
torch.save(text_model.state_dict(), "text_expert_binary.pt")
torch.save(img_model.state_dict(), "img_expert_binary.pt")
torch.save(aud_model.state_dict(), "aud_expert_binary.pt")
torch.save(meta_model.state_dict(), "meta_fusion_binary.pt")
print("\nSaved: text_expert_binary.pt img_expert_binary.pt aud_expert_binary.pt meta_fusion_binary.pt")
