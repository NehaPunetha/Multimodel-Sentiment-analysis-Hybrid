import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    vgg16, VGG16_Weights,
)
from transformers import (
    RobertaTokenizer, RobertaModel,
    BertTokenizer, BertModel,
    pipeline,
)
from transformers import ViTModel, ViTImageProcessor

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

import all_MVSA_Single as base


LABELS = ["positive", "negative", "neutral"]
MCDM_WEIGHTS = np.array([0.25, 0.25, 0.20, 0.15, 0.15])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

TEXT_BACKBONES = ["roberta-base", "bert-base-uncased"]
IMAGE_BACKBONES = ["resnet50", "vgg16", "effnet_b0", "effnet_b3", "vit_b16"]

COMBOS_TO_EVAL = [
    ("roberta-base", "resnet50"),
    ("roberta-base", "effnet_b3"),
    ("bert-base-uncased", "effnet_b0"),
    ("bert-base-uncased", "vit_b16"),
]


class TextHeadMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class ImageHeadCNN(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool1d(16)

        self.fc1 = nn.Linear(256 * 16, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        residual = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + residual

        x = self.pool(x)
        pooled = x.mean(dim=2)
        att = self.attention(pooled).unsqueeze(2)
        x = x * att

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn_fc3(self.fc3(x))))
        return self.fc4(x)


def load_text_backbone(name):
    if name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(name)
        model = RobertaModel.from_pretrained(name).to(device)
        dim = 768
    elif name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(name)
        model = BertModel.from_pretrained(name).to(device)
        dim = 768
    else:
        raise ValueError(f"Unknown text backbone: {name}")
    model.eval()
    return tokenizer, model, dim


def load_image_backbone(name):
    if name == "resnet50":
        w = ResNet50_Weights.DEFAULT
        base = resnet50(weights=w)
        preprocess = w.transforms()
        feature_extractor = nn.Sequential(*list(base.children())[:-1])  # (B,2048,1,1)
        dim = 2048

    elif name == "vgg16":
        w = VGG16_Weights.DEFAULT
        base = vgg16(weights=w)
        preprocess = w.transforms()
        # use only conv features; we'll do our own pooling+flatten
        feature_extractor = base.features  # (B,512,H',W')
        dim = 512  # after global pooling we get (B,512)

    elif name == "effnet_b0":
        w = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=w)
        preprocess = w.transforms()
        feature_extractor = nn.Sequential(*list(base.children())[:-1])
        dim = 1280

    elif name == "effnet_b3":
        w = EfficientNet_B3_Weights.DEFAULT
        base = efficientnet_b3(weights=w)
        preprocess = w.transforms()
        feature_extractor = nn.Sequential(*list(base.children())[:-1])
        dim = 1536

    elif name == "vit_b16":
        base = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        preprocess = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        feature_extractor = base
        dim = 768

    else:
        raise ValueError(f"Unknown image backbone: {name}")

    feature_extractor.eval()
    return feature_extractor.to(device), preprocess, dim


print("Loading CardiffNLP RoBERTa sentiment pipeline...")
roberta_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1
)
print("Sentiment pipeline loaded!")


def roberta_sentiment_vector(text):
    try:
        result = roberta_sentiment_pipeline(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        if "positive" in label or label == "pos":
            scores["positive"] = score
            scores["negative"] = (1 - score) / 2
            scores["neutral"] = (1 - score) / 2
        elif "negative" in label or label == "neg":
            scores["negative"] = score
            scores["positive"] = (1 - score) / 2
            scores["neutral"] = (1 - score) / 2
        else:
            scores["neutral"] = score
            scores["positive"] = (1 - score) / 2
            scores["negative"] = (1 - score) / 2
        return np.array([scores["positive"], scores["negative"], scores["neutral"]])
    except Exception as exc:
        print(f"[Warning] sentiment pipeline failed: {exc}")
        return np.array([0.33, 0.33, 0.34])


def roberta_word_level_vector(text):
    words = text.split()
    if len(words) == 0:
        return np.array([0.33, 0.33, 0.34])
    if len(words) <= 5:
        return roberta_sentiment_vector(text)
    chunk_size = 10
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    all_scores = []
    for c in chunks:
        if c.strip():
            all_scores.append(roberta_sentiment_vector(c))
    if all_scores:
        return np.mean(all_scores, axis=0)
    return np.array([0.33, 0.33, 0.34])


def get_text_embedding(text, tokenizer, backbone):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = backbone(**inputs)
        last_hidden = outputs.last_hidden_state
        cls_embedding = last_hidden[:, 0, :]
    return cls_embedding


def get_image_features(image_path, img_backbone_name, img_backbone, img_preprocess):
    if img_backbone_name == "vit_b16":
        image = Image.open(image_path).convert("RGB")
        inputs = img_preprocess(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = img_backbone(**inputs)
            feats = outputs.pooler_output  # (1,768)
        return feats
    else:
        img = Image.open(image_path).convert("RGB")
        img_tensor = img_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = img_backbone(img_tensor)  # shape depends on backbone
            # for resnet/effnet: (B,C,1,1); for vgg16 features: (B,512,H',W')
            if feats.dim() == 4:
                feats = F.adaptive_avg_pool2d(feats, (1, 1))  # (B,C,1,1)
            feats = feats.view(feats.size(0), -1)  # (B,C)
        return feats


def predict_logits_in_batches(model, input_tensor, batch_size=64):
    model.eval()
    logits_list = []
    with torch.no_grad():
        n = input_tensor.size(0)
        for i in range(0, n, batch_size):
            xb = input_tensor[i:i+batch_size].to(device)
            out = model(xb)
            logits_list.append(out.cpu())
    if logits_list:
        return torch.cat(logits_list, dim=0)
    return torch.empty((0,))


def train_simple_head(features, labels, head_type="text", input_dim=768):
    if len(labels) == 0:
        print(f"No training data for {head_type}")
        return None

    if head_type == "text":
        param_grid = {"dropout": [0.3, 0.4], "lr": [0.001, 0.002],
                      "batch_size": [32, 64], "epochs": [12]}
        HeadCls = TextHeadMLP
    else:
        param_grid = {"dropout": [0.3, 0.4], "lr": [0.0005, 0.001],
                      "batch_size": [32, 64], "epochs": [15], "weight_decay": [1e-4, 1e-5]}
        HeadCls = ImageHeadCNN

    grid = list(ParameterGrid(param_grid))[:6]

    from sklearn.utils import shuffle
    indices = np.arange(len(labels))
    indices = shuffle(indices, random_state=42)
    feats = np.array(features)[indices]
    labels_arr = np.array(labels)[indices]

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std

    split_idx = int(0.8 * len(labels_arr))
    Xtr, Xval = feats[:split_idx], feats[split_idx:]
    ytr, yval = labels_arr[:split_idx], labels_arr[split_idx:]

    from collections import Counter
    cnt = Counter(ytr)
    total = len(ytr)
    class_weights = torch.tensor(
        [total / (3 * cnt.get(c, 1)) for c in range(3)],
        dtype=torch.float32
    ).to(device)

    best_f1, best_model = 0.0, None
    for params in grid:
        dropout = params["dropout"]
        lr = params["lr"]
        bs = params["batch_size"]
        epochs = params["epochs"]
        wd = params.get("weight_decay", 1e-5)

        model = HeadCls(input_dim=input_dim, num_classes=3, dropout=dropout).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
        crit = nn.CrossEntropyLoss(weight=class_weights)

        Xtr_t = torch.from_numpy(Xtr).float()
        Xval_t = torch.from_numpy(Xval).float()
        ytr_t = torch.from_numpy(ytr).long()
        yval_t = torch.from_numpy(yval).long()
        num_batches = max(1, len(ytr) // bs)

        best_trial_f1, patience = 0.0, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(ytr))
            Xtr_s = Xtr_t[perm]
            ytr_s = ytr_t[perm]
            for b in range(num_batches):
                s = b * bs
                e = min(s + bs, len(ytr))
                xb = Xtr_s[s:e].to(device)
                yb = ytr_s[s:e].to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            model.eval()
            val_logits = predict_logits_in_batches(model, Xval_t, batch_size=bs)
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(yval, val_pred, average="weighted", zero_division=0)
            sched.step(val_f1)
            if val_f1 > best_trial_f1:
                best_trial_f1, patience = val_f1, 0
            else:
                patience += 1
            if patience >= 3:
                break

        if best_trial_f1 > best_f1:
            best_f1 = best_trial_f1
            best_model = model

    print(f"Best {head_type} head F1: {best_f1:.4f}")
    best_model.mean = mean
    best_model.std = std
    return best_model


def load_dataset_paths_and_labels(label_file, data_dir):
    ids, texts, img_paths, labels = [], [], [], []
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return ids, texts, img_paths, labels

    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[1:]

    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        idx = parts[0]
        lab = parts[1].split(",")[0].strip().lower()
        if lab not in LABELS:
            continue

        text_file = os.path.join(data_dir, f"{idx}.txt")
        if not os.path.exists(text_file):
            continue
        try:
            with open(text_file, "r", encoding="utf-8") as tf:
                text = tf.read().strip()
        except UnicodeDecodeError:
            with open(text_file, "r", encoding="latin-1") as tf:
                text = tf.read().strip()
        if not text:
            continue

        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            cand = os.path.join(data_dir, f"{idx}{ext}")
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            continue

        ids.append(idx)
        texts.append(text)
        img_paths.append(img_path)
        labels.append(LABELS.index(lab))

    print(f"Loaded {len(labels)} usable samples with text+image.")
    return ids, texts, img_paths, labels


def topsis(D, W):
    R = D / np.sqrt((D ** 2).sum(axis=0) + 1e-9)
    V = R * W
    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)
    d_pos = np.linalg.norm(V - ideal_best, axis=1)
    d_neg = np.linalg.norm(V - ideal_worst, axis=1)
    return d_neg / (d_pos + d_neg + 1e-9)


def build_decision_matrix_for_sample(text, image_path):
    D = np.zeros((3, 5))
    sent_scores = roberta_sentiment_vector(text)
    word_scores = roberta_word_level_vector(text)
    D[:, 0] = sent_scores
    D[:, 1] = word_scores
    try:
        global_scores, object_scores, scene_scores = base.image_scores(image_path)
        D[:, 2] = global_scores
        D[:, 3] = object_scores
        D[:, 4] = scene_scores
    except Exception as exc:
        print(f"[Warning] base.image_scores failed: {exc}")
        D[:, 2] = D[:, 3] = D[:, 4] = np.array([0.33, 0.33, 0.34])
    return D


def ensemble_fusion(mcdm_scores, nn_probs, temperature=1.5):
    m = np.exp(mcdm_scores / temperature)
    n = np.exp(nn_probs / temperature)
    m = m / (m.sum() + 1e-9)
    n = n / (n.sum() + 1e-9)
    return 0.4 * m + 0.6 * n


def text_head_probs(text, tokenizer, text_backbone, head):
    emb = get_text_embedding(text, tokenizer, text_backbone)
    v = emb.cpu().numpy()
    if hasattr(head, "mean"):
        v = (v - head.mean) / head.std
    vt = torch.from_numpy(v).float().to(device)
    with torch.no_grad():
        logits = head(vt)
    return F.softmax(logits, dim=1)[0].cpu().numpy()


def image_head_probs(image_path, img_name, img_backbone, img_preproc, head):
    feats = get_image_features(image_path, img_name, img_backbone, img_preproc)
    v = feats.cpu().numpy()
    if hasattr(head, "mean"):
        v = (v - head.mean) / head.std
    vt = torch.from_numpy(v).float().to(device)
    with torch.no_grad():
        logits = head(vt)
    return F.softmax(logits, dim=1)[0].cpu().numpy()


def report_metrics(y_true, y_pred, name):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%")
    print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%")


if __name__ == "__main__":
    print("=" * 80)
    print("EVALUATION: TEXT vs IMAGE vs MULTIMODAL (MCDM + NN)")
    print("=" * 80)

    ids, texts, img_paths, labels = load_dataset_paths_and_labels(
        base.LABEL_FILE, base.DATA_DIR
    )

    for t_name in TEXT_BACKBONES:
        print(f"\n=== TEXT BACKBONE: {t_name} ===")
        tokenizer, t_backbone, t_dim = load_text_backbone(t_name)
        text_features = []
        for txt in texts:
            emb = get_text_embedding(txt, tokenizer, t_backbone)
            text_features.append(emb.cpu().numpy().flatten())
        text_head = train_simple_head(text_features, labels, head_type="text", input_dim=t_dim)

        y_pred_text = []
        for txt in texts:
            probs = text_head_probs(txt, tokenizer, t_backbone, text_head)
            y_pred_text.append(np.argmax(probs))
        report_metrics(labels, y_pred_text, f"TEXT-ONLY ({t_name})")

    for i_name in IMAGE_BACKBONES:
        print(f"\n=== IMAGE BACKBONE: {i_name} ===")
        img_backbone, img_preproc, img_dim = load_image_backbone(i_name)
        img_features = []
        for p in img_paths:
            feats = get_image_features(p, i_name, img_backbone, img_preproc)
            img_features.append(feats.cpu().numpy().flatten())
        img_head = train_simple_head(img_features, labels, head_type="image", input_dim=img_dim)

        y_pred_img = []
        for p in img_paths:
            probs = image_head_probs(p, i_name, img_backbone, img_preproc, img_head)
            y_pred_img.append(np.argmax(probs))
        report_metrics(labels, y_pred_img, f"IMAGE-ONLY ({i_name})")

    print("\n" + "=" * 80)
    print("MULTIMODAL (TEXT + IMAGE) WITH MCDM + NN")
    print("=" * 80)

    for (t_name, i_name) in COMBOS_TO_EVAL:
        print(f"\n--- COMBO: Text={t_name}, Image={i_name} ---")

        tokenizer, t_backbone, t_dim = load_text_backbone(t_name)
        text_features = []
        for txt in texts:
            emb = get_text_embedding(txt, tokenizer, t_backbone)
            text_features.append(emb.cpu().numpy().flatten())
        text_head = train_simple_head(text_features, labels, head_type="text", input_dim=t_dim)

        img_backbone, img_preproc, img_dim = load_image_backbone(i_name)
        img_features = []
        for p in img_paths:
            feats = get_image_features(p, i_name, img_backbone, img_preproc)
            img_features.append(feats.cpu().numpy().flatten())
        img_head = train_simple_head(img_features, labels, head_type="image", input_dim=img_dim)

        y_pred_mm = []
        for txt, p in zip(texts, img_paths):
            D = build_decision_matrix_for_sample(txt, p)
            mcdm_scores = topsis(D, MCDM_WEIGHTS)

            probs_text = text_head_probs(txt, tokenizer, t_backbone, text_head)
            probs_img = image_head_probs(p, i_name, img_backbone, img_preproc, img_head)
            nn_probs = 0.5 * probs_text + 0.5 * probs_img

            fused = ensemble_fusion(mcdm_scores, nn_probs)
            y_pred_mm.append(np.argmax(fused))

        report_metrics(labels, y_pred_mm, f"MULTIMODAL (MCDM+NN) Text={t_name}, Image={i_name}")
