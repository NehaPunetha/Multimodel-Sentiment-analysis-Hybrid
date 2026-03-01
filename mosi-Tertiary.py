# ############################################################################ BASE CODE ##################################################################################
# import os
# import re
# import zipfile
# import pickle
# from pathlib import Path
# from collections import Counter

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import ParameterGrid
# from sklearn.utils import shuffle

# from transformers import BertTokenizer, BertModel, pipeline
# from torchvision import models, transforms
# from PIL import Image
# from scipy import ndimage

# # ============================================================
# # Reproducibility / Device
# # ============================================================

# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")

# print(f"Device: {device}")

# # ============================================================
# # MOSI paths and unzip + PKL loading
# # ============================================================

# ZIP_PATH = Path("/home/neha/Dataset/mosi.zip")
# EXTRACT_ROOT = Path("/home/neha/Dataset/mosi_extracted")
# EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

# with zipfile.ZipFile(ZIP_PATH, "r") as zf:
#     if not (EXTRACT_ROOT / "audio_dict.pkl").exists():
#         print("Extracting mosi.zip ...")
#         zf.extractall(EXTRACT_ROOT)
#     else:
#         print("PKL files already present, skipping unzip.")

# print("Extracted to:", EXTRACT_ROOT.resolve())


# def load_pkl(name: str):
#     matches = list(EXTRACT_ROOT.rglob(name))
#     if not matches:
#         raise FileNotFoundError(f"Could not find {name} under {EXTRACT_ROOT}")
#     path = matches[0]
#     print(f"Loading {path}")
#     with open(path, "rb") as f:
#         return pickle.load(f)


# audio_dict = load_pkl("audio_dict.pkl")
# visual_dict = load_pkl("processed_visual_dict.pkl")
# text_emb_dict = load_pkl("text_emb.pkl")
# label_dict = load_pkl("label_dict.pkl")

# print("audio_dict:", type(audio_dict), len(audio_dict))
# print("visual_dict:", type(visual_dict), len(visual_dict))
# print(
#     "text_emb_dict:", type(text_emb_dict), len(text_emb_dict),
#     "sample shape:", next(iter(text_emb_dict.values())).shape
# )

# # ============================================================
# # Label mapping (3-class sentiment)
# # ============================================================

# LABELS = ["negative", "neutral", "positive"]
# LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


# def encode_label(raw):
#     label_str = raw.strip().lower()
#     if label_str not in LABEL_MAP:
#         raise ValueError(f"Unknown label: {raw}")
#     return LABEL_MAP[label_str]

# # ============================================================
# # Simple ID-based train/val/test split
# # ============================================================

# all_ids = sorted(
#     uid for uid in text_emb_dict.keys()
#     if uid in audio_dict and uid in visual_dict and uid in label_dict
# )
# all_ids = np.array(all_ids)
# all_ids = shuffle(all_ids, random_state=SEED)

# n = len(all_ids)
# n_train = int(0.7 * n)
# n_val = int(0.1 * n)
# train_ids = all_ids[:n_train]
# val_ids = all_ids[n_train:n_train + n_val]
# test_ids = all_ids[n_train + n_val:]

# print(f"Total IDs: {n} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# # ============================================================
# # BERT backbone (not used for training here, but kept)
# # ============================================================

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert = BertModel.from_pretrained("bert-base-uncased").to(device)
# bert.eval()


# def get_text_embedding(text: str):
#     inputs = tokenizer(
#         text[:512],
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = bert(**inputs)
#     token_embeddings = outputs.last_hidden_state
#     attention_mask = inputs["attention_mask"]
#     return token_embeddings, attention_mask

# # ============================================================
# # CNN backbone for visual heuristics (not used in MLP training)
# # ============================================================

# cnn = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
# num_features_cnn = cnn.classifier.in_features
# cnn.classifier = nn.Identity()
# for param in cnn.parameters():
#     param.requires_grad = True
# for param in cnn.features.denseblock4.parameters():
#     param.requires_grad = True
# for param in cnn.features.norm5.parameters():
#     param.requires_grad = True
# cnn = cnn.to(device)
# cnn.train()

# img_transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ]
# )

# # ============================================================
# # HuggingFace sentiment pipelines (for text SAW)
# # ============================================================

# try:
#     sentiment_pipeline = pipeline(
#         "sentiment-analysis",
#         model="cardiffnlp/twitter-roberta-base-sentiment",
#     )
# except Exception:
#     sentiment_pipeline = pipeline(
#         "sentiment-analysis",
#         model="distilbert-base-uncased-finetuned-sst-2-english",
#     )

# try:
#     fineweb_pipeline = pipeline(
#         "sentiment-analysis",
#         model="michellejieli/NLTK_based-twitter-sentiment-analysis",
#     )
# except Exception:
#     fineweb_pipeline = None

# # ============================================================
# # Lexicons / helpers
# # ============================================================

# positive_words = {
#     "good", "great", "excellent", "amazing", "love", "best", "wonderful",
#     "fantastic", "beautiful", "awesome", "perfect", "happy", "joy", "glad",
#     "brilliant", "outstanding", "superb", "terrific", "lovely", "exceptional",
#     "gorgeous", "divine", "marvelous", "splendid", "magnificent", "impressive",
#     "delightful", "enjoyable",
# }

# negative_words = {
#     "bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting",
#     "ugly", "poor", "sad", "angry", "disappointed", "pathetic", "useless",
#     "dumb", "stupid", "sucks", "gross", "weak", "fail", "failure",
#     "disappointing", "annoying", "frustrating", "mediocre", "atrocious",
#     "dreadful", "abysmal", "repulsive", "appalling",
# }

# negation_words = {
#     "no", "not", "neither", "nor", "never", "nobody", "nothing", "n't",
#     "couldn't", "shouldn't", "wouldn't", "don't", "doesn't", "didn't",
#     "won't", "can't", "isn't",
# }


# def split_sentences(text: str):
#     sentences = re.split(r"[.!?]+", text)
#     return [s.strip() for s in sentences if s.strip()]

# # ============================================================
# # SAW utilities
# # ============================================================

# def saw(decision_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
#     denom = decision_matrix.max(axis=0)
#     denom[denom == 0] = 1.0
#     norm_matrix = decision_matrix / denom
#     return norm_matrix @ weights


# def saw_normalize(D: np.ndarray) -> np.ndarray:
#     maxs = D.max(axis=0)
#     mins = D.min(axis=0)
#     diff = maxs - mins
#     diff[diff == 0] = 1.0
#     return (D - mins) / diff


# def final_text_label(pos: float, neg: float, neu: float) -> str:
#     scores = np.array([pos, neg, neu])
#     dominant = np.argmax(scores)
#     sorted_scores = np.sort(scores)[::-1]
#     if sorted_scores[0] - sorted_scores[1] > 0.10:
#         return ["positive", "negative", "neutral"][dominant]
#     return ["positive", "negative", "neutral"][dominant]


# def text_saw_classifier(word_scores: np.ndarray, sent_scores: np.ndarray) -> str:
#     D = np.vstack([word_scores, sent_scores])
#     R = saw_normalize(D)
#     W = np.array([0.5, 0.5])
#     saw_scores = R.T @ W
#     pos, neg, neu = saw_scores
#     return final_text_label(pos, neg, neu)

# # ============================================================
# # Text / image / audio scoring (heuristics for SAW)
# # ============================================================

# def text_scores(text: str):
#     try:
#         if isinstance(text, bytes):
#             text = text.decode("utf-8", errors="ignore")
#         text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
#         text = re.sub(r"\s+", " ", text).strip()

#         scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

#         result = sentiment_pipeline(text[:512])[0]
#         label = result["label"].lower()
#         confidence = result["score"]

#         if "positive" in label or label == "pos":
#             scores["positive"] += confidence * 0.4
#             scores["negative"] += (1 - confidence) * 0.12
#             scores["neutral"] += (1 - confidence) * 0.28
#         elif "negative" in label or label == "neg":
#             scores["negative"] += confidence * 0.4
#             scores["positive"] += (1 - confidence) * 0.12
#             scores["neutral"] += (1 - confidence) * 0.28
#         else:
#             scores["neutral"] += confidence * 0.4
#             scores["positive"] += (1 - confidence) * 0.3
#             scores["negative"] += (1 - confidence) * 0.3

#         if fineweb_pipeline is not None:
#             try:
#                 result2 = fineweb_pipeline(text[:512])[0]
#                 label2 = result2["label"].lower()
#                 confidence2 = result2["score"]
#                 if "positive" in label2:
#                     scores["positive"] += confidence2 * 0.15
#                 elif "negative" in label2:
#                     scores["negative"] += confidence2 * 0.15
#                 else:
#                     scores["neutral"] += confidence2 * 0.15
#             except Exception:
#                 pass

#         sentences = split_sentences(text)
#         for sentence in sentences:
#             sent_lower = sentence.lower()
#             tokens = sent_lower.split()
#             has_negation = any(neg in tokens for neg in negation_words)
#             pos_count = sum(1 for w in positive_words if w in sent_lower)
#             neg_count = sum(1 for w in negative_words if w in sent_lower)
#             if has_negation:
#                 if pos_count > 0:
#                     scores["negative"] += pos_count * 0.08
#                 if neg_count > 0:
#                     scores["positive"] += neg_count * 0.08
#             else:
#                 if pos_count > neg_count:
#                     scores["positive"] += (pos_count - neg_count) * 0.08
#                 elif neg_count > pos_count:
#                     scores["negative"] += (neg_count - pos_count) * 0.08

#         exclamation_count = text.count("!")
#         caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
#         if exclamation_count > 0:
#             scores["positive"] += exclamation_count * 0.05
#         if caps_ratio > 0.25:
#             scores["positive"] += 0.1

#         total = sum(scores.values())
#         if total > 0:
#             word_probs = np.array(
#                 [
#                     scores["positive"] / total,
#                     scores["negative"] / total,
#                     scores["neutral"] / total,
#                 ]
#             )
#         else:
#             word_probs = np.array([0.33, 0.33, 0.34])

#         sent_probs = word_probs * 0.9 + np.array([0.33, 0.33, 0.34]) * 0.1
#         sent_probs = sent_probs / sent_probs.sum()

#     except Exception as e:
#         print(f"Error in sentiment analysis (text_scores): {e}")
#         word_probs = np.array([0.33, 0.33, 0.34])
#         sent_probs = np.array([0.33, 0.33, 0.34])

#     return word_probs, sent_probs


# def image_scores_from_visual(visual_seq: np.ndarray):
#     if visual_seq.ndim != 2:
#         return (
#             np.array([0.33, 0.33, 0.34]),
#             np.array([0.33, 0.33, 0.34]),
#             np.array([0.33, 0.33, 0.34]),
#         )
#     mean_vec = visual_seq.mean(axis=0)
#     std_vec = visual_seq.std(axis=0)
#     energy = float(np.linalg.norm(mean_vec))
#     variability = float(np.linalg.norm(std_vec))
#     pos_score = max(0.0, energy * 0.4 + variability * 0.2)
#     neg_score = max(0.0, (1.0 / (1.0 + energy)) * 0.6 + variability * 0.1)
#     neu_score = max(0.0, 1.0 - (abs(energy - variability) * 0.1))
#     total = pos_score + neg_score + neu_score
#     if total > 0:
#         adjusted_scores = np.array(
#             [pos_score / total, neg_score / total, neu_score / total]
#         )
#     else:
#         adjusted_scores = np.array([0.33, 0.33, 0.34])
#     obj_scores = adjusted_scores * 0.9 + 0.05
#     scene_scores = adjusted_scores * 1.05
#     scene_scores = scene_scores / scene_scores.sum()
#     return adjusted_scores, obj_scores, scene_scores


# def audio_scores(audio_seq: np.ndarray):
#     if audio_seq.ndim != 2:
#         return np.array([0.33, 0.33, 0.34])
#     mean_vec = audio_seq.mean(axis=0)
#     std_vec = audio_seq.std(axis=0)
#     energy = float(np.linalg.norm(mean_vec))
#     dynamics = float(np.linalg.norm(std_vec))
#     pos_score = max(0.0, energy * 0.4 + dynamics * 0.2)
#     neg_score = max(0.0, (1.0 / (1.0 + energy)) * 0.5 + (1.0 / (1.0 + dynamics)) * 0.2)
#     neu_score = max(0.0, 1.0 - abs(energy - dynamics) * 0.1)
#     total = pos_score + neg_score + neu_score
#     if total > 0:
#         probs = np.array(
#             [pos_score / total, neg_score / total, neu_score / total]
#         )
#     else:
#         probs = np.array([0.33, 0.33, 0.34])
#     return probs

# # ============================================================
# # NN heads
# # ============================================================

# ## Base
# # class TextSentimentMLP(nn.Module):
# #     def __init__(self, input_dim=768, hidden_dim=256, num_classes=3, dropout=0.4):
# #         super().__init__()
# #         self.fc1 = nn.Linear(input_dim, hidden_dim)
# #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
# #         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
# #         self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, x):
# #         x = self.fc1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
# #         x = self.fc2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
# #         logits = self.fc_out(x)
# #         return logits

# class TextSentimentMLP(nn.Module):
#     def __init__(self, input_dim=768, num_classes=3, hidden_dim=(512, 256, 128), dropout=0.5):
#         super().__init__()
#         layers = []
#         prev = input_dim
#         for h in hidden_dim:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.BatchNorm1d(h))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(dropout))
#             prev = h
#         self.feature = nn.Sequential(*layers)
#         self.fc_out = nn.Linear(prev, num_classes)

#     def forward(self, x):
#         x = self.feature(x)
#         return self.fc_out(x)

# # class ImageSentimentMLP(nn.Module):
# #     def __init__(self, input_dim, num_classes=3, dropout=0.5):
# #         super().__init__()
# #         self.fc1 = nn.Linear(input_dim, 1024)
# #         self.bn1 = nn.BatchNorm1d(1024)
# #         self.fc2 = nn.Linear(1024, 512)
# #         self.bn2 = nn.BatchNorm1d(512)
# #         self.fc3 = nn.Linear(512, 256)
# #         self.bn3 = nn.BatchNorm1d(256)
# #         self.fc_out = nn.Linear(256, num_classes)
# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, x):
# #         x = self.fc1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
# #         x = self.fc2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
# #         x = self.fc3(x); x = self.bn3(x); x = F.relu(x); x = self.dropout(x)
# #         logits = self.fc_out(x)
# #         return logits

# class ImageSentimentMLP(nn.Module):
#     def __init__(self, input_dim, num_classes=3, hidden_dims=(256, 128), dropout=0.4):
#         super().__init__()
#         layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.BatchNorm1d(h))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(dropout))
#             prev = h
#         self.feature = nn.Sequential(*layers)
#         self.fc_out = nn.Linear(prev, num_classes)

#     def forward(self, x):
#         x = self.feature(x)
#         return self.fc_out(x)


# # class AudioSentimentMLP(nn.Module):
# #     def __init__(self, input_dim, num_classes=3, hidden_dim=128, dropout=0.4):
# #         super().__init__()
# #         self.fc1 = nn.Linear(input_dim, hidden_dim)
# #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
# #         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
# #         self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, x):
# #         x = self.fc1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
# #         x = self.fc2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
# #         logits = self.fc_out(x)
# #         return logits

# class AudioSentimentMLP(nn.Module):
#     def __init__(self, input_dim, num_classes=3, hidden_dim=(128, 64), dropout=0.4):
#         super().__init__()
#         layers = []
#         prev = input_dim
#         for h in hidden_dim:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.BatchNorm1d(h))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(dropout))
#             prev = h
#         self.feature = nn.Sequential(*layers)
#         self.fc_out = nn.Linear(prev, num_classes)

#     def forward(self, x):
#         x = self.feature(x)
#         return self.fc_out(x)

# # ============================================================
# # MCDM methods
# # ============================================================

# def text_sentiment_tag_saw_only_from_embedding(text_vec_cls: np.ndarray):
#     v = text_vec_cls
#     energy = float(np.linalg.norm(v))
#     pos = energy * 0.4
#     neg = (1.0 / (1.0 + energy)) * 0.4
#     neu = 1.0
#     total = pos + neg + neu
#     scores = np.array([pos / total, neg / total, neu / total])
#     return LABELS[int(np.argmax(scores))]


# METHODS = ["SAW", "TOPSIS", "RAFSI", "TODIM", "MARCOS"]


# def minmax_norm(D):
#     mn, mx = D.min(axis=0), D.max(axis=0)
#     diff = np.where(mx - mn == 0, 1, mx - mn)
#     return (D - mn) / diff


# def mcd_saw(D, W):
#     return minmax_norm(D) @ W


# def topsis(D, W):
#     R = D / np.sqrt((D ** 2).sum(axis=0))
#     V = R * W
#     d_pos = np.linalg.norm(V - V.max(axis=0), axis=1)
#     d_neg = np.linalg.norm(V - V.min(axis=0), axis=1)
#     return d_neg / (d_pos + d_neg + 1e-9)


# def rafsi(D, W):
#     R = minmax_norm(D)
#     ranks = R.argsort(axis=0).argsort(axis=0)
#     return (ranks * W).sum(axis=1)


# def todim(D, W, theta=1.0):
#     R = minmax_norm(D)
#     n = R.shape[0]
#     score = np.zeros(n)
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             diff = R[i] - R[j]
#             score[i] += (
#                 np.sum(W * np.maximum(diff, 0))
#                 - np.sum(W * np.maximum(-diff, 0)) / theta
#             )
#     return score


# def marcos(D, W):
#     S = minmax_norm(D) @ W
#     return S / (S.sum() + 1e-9)


# def method_scores(D, W, method):
#     if method == "SAW":
#         return mcd_saw(D, W)
#     if method == "TOPSIS":
#         return topsis(D, W)
#     if method == "RAFSI":
#         return rafsi(D, W)
#     if method == "TODIM":
#         return todim(D, W)
#     if method == "MARCOS":
#         return marcos(D, W)
#     raise ValueError(f"Unknown method: {method}")

# # ============================================================
# # Evaluation helper
# # ============================================================

# def evaluate(y_true, y_pred, name):
#     print(f"\n{name}")
#     print("-" * len(name))
#     print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
#     print(
#         f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
#     )
#     print(
#         f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
#     )
#     print(
#         f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
#     )

# # ============================================================
# # Original simple feature builder
# # ============================================================

# def build_split_features(id_list):
#     text_emb_list = []
#     img_feat_list = []
#     audio_feat_list = []
#     labels_list = []
#     for uid in id_list:
#         a = np.asarray(audio_dict[uid], dtype=np.float32)
#         v = np.asarray(visual_dict[uid], dtype=np.float32)
#         t = np.asarray(text_emb_dict[uid], dtype=np.float32)
#         if a.ndim != 2 or v.ndim != 2 or t.ndim != 2:
#             continue
#         label_s = label_dict[uid]
#         try:
#             y = encode_label(label_s)
#         except ValueError:
#             continue
#         text_vec = t[0]
#         img_vec = v.mean(axis=0)
#         audio_vec = a.mean(axis=0)
#         text_emb_list.append(text_vec)
#         img_feat_list.append(img_vec)
#         audio_feat_list.append(audio_vec)
#         labels_list.append(y)
#     return (
#         np.array(text_emb_list, dtype=np.float32),
#         np.array(img_feat_list, dtype=np.float32),
#         np.array(audio_feat_list, dtype=np.float32),
#         np.array(labels_list, dtype=np.int64),
#     )

# # ============================================================
# # Flexible feature builder for ablation
# # ============================================================

# def build_split_features_for_setup(
#     id_list,
#     use_text=True,
#     use_image=True,
#     use_audio=True,
#     text_emb_source=None,
# ):
#     text_emb_list = []
#     img_feat_list = []
#     audio_feat_list = []
#     labels_list = []

#     text_src = text_emb_dict if text_emb_source is None else text_emb_source

#     for uid in id_list:
#         if uid not in label_dict:
#             continue

#         # text
#         if use_text:
#             if uid not in text_src:
#                 continue
#             t = np.asarray(text_src[uid], dtype=np.float32)
#             if t.ndim == 2:
#                 text_vec = t[0]
#             else:
#                 text_vec = t
#         else:
#             text_vec = np.zeros(1, dtype=np.float32)

#         # image
#         if use_image:
#             if uid not in visual_dict:
#                 continue
#             v = np.asarray(visual_dict[uid], dtype=np.float32)
#             if v.ndim != 2:
#                 continue
#             img_vec = v.mean(axis=0)
#         else:
#             img_vec = np.zeros(1, dtype=np.float32)

#         # audio
#         if use_audio:
#             if uid not in audio_dict:
#                 continue
#             a = np.asarray(audio_dict[uid], dtype=np.float32)
#             if a.ndim != 2:
#                 continue
#             audio_vec = a.mean(axis=0)
#         else:
#             audio_vec = np.zeros(1, dtype=np.float32)

#         label_s = label_dict[uid]
#         try:
#             y = encode_label(label_s)
#         except ValueError:
#             continue

#         text_emb_list.append(text_vec)
#         img_feat_list.append(img_vec)
#         audio_feat_list.append(audio_vec)
#         labels_list.append(y)

#     if not labels_list:
#         return None

#     return (
#         np.array(text_emb_list, dtype=np.float32),
#         np.array(img_feat_list, dtype=np.float32),
#         np.array(audio_feat_list, dtype=np.float32),
#         np.array(labels_list, dtype=np.int64),
#     )

# # ============================================================
# # Training (hyperparameter search) for text/image/audio
# # ============================================================

# def train_neural_networks_mosi(
#     text_embeddings_np,
#     image_features_np,
#     audio_features_np,
#     labels_arr,
#     param_grid=None,
#     max_trials=5,
# ):
#     if len(labels_arr) == 0:
#         print("No training data, skipping training.")
#         return None, None, None

#     if param_grid is None:
#         param_grid = {
#             "dropout": [0.3, 0.4, 0.5],
#             "lr": [0.0005, 0.001, 0.005, 0.01],
#             "batch_size": [32, 64, 128],
#             "epochs": [30, 50, 70],
#         }

#     grid = list(ParameterGrid(param_grid))[:max_trials]

#     img_mean = image_features_np.mean(axis=0, keepdims=True)
#     img_std = image_features_np.std(axis=0, keepdims=True) + 1e-6
#     image_features_np = (image_features_np - img_mean) / img_std

#     aud_mean = audio_features_np.mean(axis=0, keepdims=True)
#     aud_std = audio_features_np.std(axis=0, keepdims=True) + 1e-6
#     audio_features_np = (audio_features_np - aud_mean) / aud_std

#     best_text_acc = 0.0
#     best_image_acc = 0.0
#     best_audio_acc = 0.0
#     best_params = None
#     best_text_model = None
#     best_image_model = None
#     best_audio_model = None

#     label_counts = Counter(labels_arr)
#     total = len(labels_arr)
#     class_weights = torch.tensor(
#         [total / (3 * label_counts.get(c, 1)) for c in range(3)],
#         dtype=torch.float32,
#     ).to(device)

#     text_dim = text_embeddings_np.shape[1]
#     img_dim = image_features_np.shape[1]
#     aud_dim = audio_features_np.shape[1]

#     for params in grid:
#         text_model = TextSentimentMLP(
#             input_dim=text_dim,
#             hidden_dim=(512, 256, 128),
#             num_classes=3,
#             dropout=params["dropout"],
#         ).to(device)

#         image_model = ImageSentimentMLP(
#             input_dim=img_dim,
#             num_classes=3,
#             dropout=params["dropout"],
#         ).to(device)

#         audio_model = AudioSentimentMLP(
#             input_dim=aud_dim,
#             num_classes=3,
#             hidden_dim=(128, 64),
#             dropout=params["dropout"],
#         ).to(device)

#         text_optimizer = optim.Adam(
#             text_model.parameters(), lr=params["lr"], weight_decay=3e-5
#         )
#         image_optimizer = optim.Adam(
#             image_model.parameters(), lr=params["lr"], weight_decay=3e-5
#         )
#         audio_optimizer = optim.Adam(
#             audio_model.parameters(), lr=params["lr"], weight_decay=3e-5
#         )

#         batch_size = params["batch_size"]
#         epochs = params["epochs"]

#         text_tensor = torch.from_numpy(text_embeddings_np).float().to(device)
#         image_tensor = torch.from_numpy(image_features_np).float().to(device)
#         audio_tensor = torch.from_numpy(audio_features_np).float().to(device)
#         labels_tensor = torch.from_numpy(labels_arr).long().to(device)

#         criterion = nn.CrossEntropyLoss(weight=class_weights)

#         num_samples = len(labels_arr)
#         num_batches = max(1, (num_samples + batch_size - 1) // batch_size)

#         best_val_loss = float('inf')
#         patience = 5
#         patience_counter = 0
#         for epoch in range(epochs):
#             text_model.train()
#             image_model.train()
#             audio_model.train()

#             perm = torch.randperm(num_samples)
#             text_epoch = text_tensor[perm]
#             image_epoch = image_tensor[perm]
#             audio_epoch = audio_tensor[perm]
#             labels_epoch = labels_tensor[perm]

#             for b in range(num_batches):
#                 start = b * batch_size
#                 end = min(start + batch_size, num_samples)

#                 xb_text = text_epoch[start:end]
#                 xb_image = image_epoch[start:end]
#                 xb_audio = audio_epoch[start:end]
#                 yb = labels_epoch[start:end]

#                 text_optimizer.zero_grad()
#                 logits_text = text_model(xb_text)
#                 loss_text = criterion(logits_text, yb)
#                 loss_text.backward()
#                 nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
#                 text_optimizer.step()

#                 image_optimizer.zero_grad()
#                 logits_image = image_model(xb_image)
#                 loss_image = criterion(logits_image, yb)
#                 loss_image.backward()
#                 nn.utils.clip_grad_norm_(image_model.parameters(), max_norm=1.0)
#                 image_optimizer.step()

#                 audio_optimizer.zero_grad()
#                 logits_audio = audio_model(xb_audio)
#                 loss_audio = criterion(logits_audio, yb)
#                 loss_audio.backward()
#                 nn.utils.clip_grad_norm_(audio_model.parameters(), max_norm=1.0)
#                 audio_optimizer.step()

#             # Early stopping: use a split of train as pseudo-val
#             text_model.eval()
#             with torch.no_grad():
#                 val_loss = criterion(text_model(text_tensor), labels_tensor).item()
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break

#         text_model.eval()
#         image_model.eval()
#         audio_model.eval()
#         with torch.no_grad():
#             logits_text = text_model(text_tensor)
#             preds_text = torch.argmax(logits_text, dim=1).cpu().numpy()
#             text_acc = (preds_text == labels_arr).mean()

#             logits_image = image_model(image_tensor)
#             preds_image = torch.argmax(logits_image, dim=1).cpu().numpy()
#             image_acc = (preds_image == labels_arr).mean()

#             logits_audio = audio_model(audio_tensor)
#             preds_audio = torch.argmax(logits_audio, dim=1).cpu().numpy()
#             audio_acc = (preds_audio == labels_arr).mean()

#         print(
#             f"Params: {params} | Text train-Acc: {text_acc:.4f} | "
#             f"Image train-Acc: {image_acc:.4f} | Audio train-Acc: {audio_acc:.4f}"
#         )

#         if (
#             text_acc > best_text_acc
#             or image_acc > best_image_acc
#             or audio_acc > best_audio_acc
#         ):
#             best_text_acc = text_acc
#             best_image_acc = image_acc
#             best_audio_acc = audio_acc
#             best_params = params
#             best_text_model = text_model
#             best_image_model = image_model
#             best_audio_model = audio_model

#     print(f"\nBest Params (train-set sanity): {best_params}")
#     print(f"Best Text (train) Acc : {best_text_acc:.4f}")
#     print(f"Best Image (train) Acc: {best_image_acc:.4f}")
#     print(f"Best Audio (train) Acc: {best_audio_acc:.4f}")

#     best_image_model.img_mean = img_mean
#     best_image_model.img_std = img_std
#     best_audio_model.aud_mean = aud_mean
#     best_audio_model.aud_std = aud_std

#     return best_text_model, best_image_model, best_audio_model

# # ============================================================
# # Run one ablation setup
# # ============================================================

# def run_setup(
#     setup_name,
#     use_text=True,
#     use_image=True,
#     use_audio=True,
#     text_emb_source=None,
#     param_grid=None,
# ):
#     print("\n" + "=" * 60)
#     print(f"SETUP: {setup_name}")
#     print("=" * 60)

#     train_feats = build_split_features_for_setup(
#         train_ids,
#         use_text=use_text,
#         use_image=use_image,
#         use_audio=use_audio,
#         text_emb_source=text_emb_source,
#     )
#     val_feats = build_split_features_for_setup(
#         val_ids,
#         use_text=use_text,
#         use_image=use_image,
#         use_audio=use_audio,
#         text_emb_source=text_emb_source,
#     )
#     test_feats = build_split_features_for_setup(
#         test_ids,
#         use_text=use_text,
#         use_image=use_image,
#         use_audio=use_audio,
#         text_emb_source=text_emb_source,
#     )

#     if train_feats is None or val_feats is None or test_feats is None:
#         print("No data for this setup, skipping.")
#         return

#     text_train, img_train, aud_train, y_train = train_feats
#     text_val, img_val, aud_val, y_val = val_feats
#     text_test, img_test, aud_test, y_test = test_feats

#     print(f"Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)}")

#     text_nn, img_nn, aud_nn = train_neural_networks_mosi(
#         text_train, img_train, aud_train, y_train, param_grid=param_grid, max_trials=3
#     )

#     img_mean = getattr(img_nn, "img_mean", None)
#     img_std = getattr(img_nn, "img_std", None)
#     aud_mean = getattr(aud_nn, "aud_mean", None)
#     aud_std = getattr(aud_nn, "aud_std", None)

#     y_true = []
#     text_preds_saw = []
#     text_preds_nn = []
#     img_preds_saw = []
#     img_preds_nn = []
#     aud_preds_saw = []
#     aud_preds_nn = []
#     fused_nn_preds = []
#     mcdm_multi_preds = {m: [] for m in METHODS}

#     text_src = text_emb_dict if text_emb_source is None else text_emb_source

#     for uid in test_ids:
#         if uid not in label_dict:
#             continue

#         # build per-uid vectors consistent with training dims
#         if use_text:
#             if uid not in text_src:
#                 continue
#             t_arr = np.asarray(text_src[uid], dtype=np.float32)
#             if t_arr.ndim != 2:
#                 continue
#             t_vec = t_arr[0]
#         else:
#             t_vec = np.zeros(text_train.shape[1], dtype=np.float32)

#         if use_image:
#             if uid not in visual_dict:
#                 continue
#             v_arr = np.asarray(visual_dict[uid], dtype=np.float32)
#             if v_arr.ndim != 2:
#                 continue
#             v_vec = v_arr.mean(axis=0)
#         else:
#             v_vec = np.zeros(img_train.shape[1], dtype=np.float32)

#         if use_audio:
#             if uid not in audio_dict:
#                 continue
#             a_arr = np.asarray(audio_dict[uid], dtype=np.float32)
#             if a_arr.ndim != 2:
#                 continue
#             a_vec = a_arr.mean(axis=0)
#         else:
#             a_vec = np.zeros(aud_train.shape[1], dtype=np.float32)

#         label_s = label_dict[uid]
#         try:
#             y = encode_label(label_s)
#         except ValueError:
#             continue

#         # Text SAW-only
#         if use_text:
#             t_pred_saw = text_sentiment_tag_saw_only_from_embedding(t_vec)
#         else:
#             t_pred_saw = "neutral"
#         text_preds_saw.append(t_pred_saw)

#         # Text NN
#         tv = torch.from_numpy(t_vec[None, :]).float().to(device)
#         with torch.no_grad():
#             logits_t = text_nn(tv)
#             probs_t = F.softmax(logits_t, dim=1)[0].cpu().numpy()
#         text_preds_nn.append(LABELS[int(np.argmax(probs_t))])

#         # Image SAW-only (make a pseudo sequence)
#         if use_image:
#             g, o, s = image_scores_from_visual(
#                 np.tile(v_vec[None, :], (3, 1))
#             )
#             D_img = np.array(
#                 [
#                     [g[0], o[0], s[0]],
#                     [g[1], o[1], s[1]],
#                     [g[2], o[2], s[2]],
#                 ]
#             )
#             W_img = np.array([0.6, 0.25, 0.15])
#             scores_img_saw = saw(D_img, W_img)
#             img_preds_saw.append(LABELS[int(np.argmax(scores_img_saw))])
#         else:
#             img_preds_saw.append("neutral")

#         # Image NN
#         if img_mean is not None and img_std is not None:
#             img_norm = (v_vec[None, :] - img_mean) / img_std
#         else:
#             img_norm = v_vec[None, :]
#         iv = torch.from_numpy(img_norm).float().to(device)
#         with torch.no_grad():
#             logits_i = img_nn(iv)
#             probs_i = F.softmax(logits_i, dim=1)[0].cpu().numpy()
#         img_preds_nn.append(LABELS[int(np.argmax(probs_i))])

#         # Audio SAW-only
#         if use_audio:
#             probs_a_saw = audio_scores(
#                 np.tile(a_vec[None, :], (3, 1))
#             )
#             aud_preds_saw.append(LABELS[int(np.argmax(probs_a_saw))])
#         else:
#             aud_preds_saw.append("neutral")

#         # Audio NN
#         if aud_mean is not None and aud_std is not None:
#             aud_norm = (a_vec[None, :] - aud_mean) / aud_std
#         else:
#             aud_norm = a_vec[None, :]
#         av = torch.from_numpy(aud_norm).float().to(device)
#         with torch.no_grad():
#             logits_a = aud_nn(av)
#             probs_a = F.softmax(logits_a, dim=1)[0].cpu().numpy()
#         aud_preds_nn.append(LABELS[int(np.argmax(probs_a))])

#         # Improved fusion: learnable weights (here, softmaxed static weights for demonstration)
#         fusion_weights = np.array([1.0, 0.8, 0.8])
#         fusion_weights = np.exp(fusion_weights) / np.sum(np.exp(fusion_weights))
#         fused_probs_nn = fusion_weights[0] * probs_t + fusion_weights[1] * probs_i + fusion_weights[2] * probs_a
#         fused_probs_nn = fused_probs_nn / fused_probs_nn.sum()
#         fused_nn_preds.append(LABELS[int(np.argmax(fused_probs_nn))])

#         D_multi = np.stack([probs_t, probs_i, probs_a], axis=1)
#         W_multi = np.array([0.4, 0.3, 0.3])

#         for m in METHODS:
#             m_scores = method_scores(D_multi, W_multi, m)
#             mcdm_multi_preds[m].append(LABELS[int(np.argmax(m_scores))])

#         y_true.append(LABELS[y])

#     print("\nTest samples used:", len(y_true))

#     evaluate(y_true, text_preds_saw, f"{setup_name} - TEXT SAW-only")
#     evaluate(y_true, text_preds_nn, f"{setup_name} - TEXT NN")
#     evaluate(y_true, img_preds_saw, f"{setup_name} - IMAGE SAW-only")
#     evaluate(y_true, img_preds_nn, f"{setup_name} - IMAGE NN")
#     evaluate(y_true, aud_preds_saw, f"{setup_name} - AUDIO SAW-only")
#     evaluate(y_true, aud_preds_nn, f"{setup_name} - AUDIO NN")
#     evaluate(y_true, fused_nn_preds, f"{setup_name} - MULTIMODAL NN (avg probs)")

#     for m in METHODS:
#         evaluate(y_true, mcdm_multi_preds[m], f"{setup_name} - MULTIMODAL {m}")

# # ============================================================
# # MAIN: run ablation study
# # ============================================================

# if __name__ == "__main__":
#     param_grid = {
#         "dropout": [0.5],
#         "lr": [0.01],
#         "batch_size": [64],
#         "epochs": [50],
#     }

#     # Baseline: all three modalities
#     run_setup(
#         "TEXT+IMAGE+AUDIO (baseline)",
#         use_text=True,
#         use_image=True,
#         use_audio=True,
#         text_emb_source=None,
#         param_grid=param_grid,
#     )

#     # Text-only
#     run_setup(
#         "TEXT only",
#         use_text=True,
#         use_image=False,
#         use_audio=False,
#         text_emb_source=None,
#         param_grid=param_grid,
#     )

#     # Image-only
#     run_setup(
#         "IMAGE only",
#         use_text=False,
#         use_image=True,
#         use_audio=False,
#         text_emb_source=None,
#         param_grid=param_grid,
#     )

#     # Audio-only
#     run_setup(
#         "AUDIO only",
#         use_text=False,
#         use_image=False,
#         use_audio=True,
#         text_emb_source=None,
#         param_grid=param_grid,
#     )

#     # Text+Audio
#     run_setup(
#         "TEXT+AUDIO",
#         use_text=True,
#         use_image=False,
#         use_audio=True,
#         text_emb_source=None,
#         param_grid=param_grid,
#     )

#     # Text+Image
#     run_setup(
#         "TEXT+IMAGE",
#         use_text=True,
#         use_image=True,
#         use_audio=False,
#         text_emb_source=None,
#         param_grid=param_grid,
#     )


############################################################################ BASE CODE ##################################################################################
import os
import re
import zipfile
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle

from transformers import BertTokenizer, BertModel, pipeline
from torchvision import models, transforms
from PIL import Image
from scipy import ndimage

# ============================================================
# Reproducibility / Device
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# ============================================================
# MOSI paths and unzip + PKL loading
# ============================================================

ZIP_PATH = Path("/home/neha/Dataset/mosi.zip")
EXTRACT_ROOT = Path("/home/neha/Dataset/mosi_extracted")
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    if not (EXTRACT_ROOT / "audio_dict.pkl").exists():
        print("Extracting mosi.zip ...")
        zf.extractall(EXTRACT_ROOT)
    else:
        print("PKL files already present, skipping unzip.")

print("Extracted to:", EXTRACT_ROOT.resolve())


def load_pkl(name: str):
    matches = list(EXTRACT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {EXTRACT_ROOT}")
    path = matches[0]
    print(f"Loading {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


audio_dict = load_pkl("audio_dict.pkl")
visual_dict = load_pkl("processed_visual_dict.pkl")
text_emb_dict = load_pkl("text_emb.pkl")
label_dict = load_pkl("label_dict.pkl")

print("audio_dict:", type(audio_dict), len(audio_dict))
print("visual_dict:", type(visual_dict), len(visual_dict))
print(
    "text_emb_dict:", type(text_emb_dict), len(text_emb_dict),
    "sample shape:", next(iter(text_emb_dict.values())).shape
)

# ============================================================
# Label mapping (3-class sentiment)
# ============================================================

LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def encode_label(raw):
    label_str = raw.strip().lower()
    if label_str not in LABEL_MAP:
        raise ValueError(f"Unknown label: {raw}")
    return LABEL_MAP[label_str]

# ============================================================
# Simple ID-based train/val/test split
# ============================================================

all_ids = sorted(
    uid for uid in text_emb_dict.keys()
    if uid in audio_dict and uid in visual_dict and uid in label_dict
)
all_ids = np.array(all_ids)
all_ids = shuffle(all_ids, random_state=SEED)

n = len(all_ids)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
train_ids = all_ids[:n_train]
val_ids = all_ids[n_train:n_train + n_val]
test_ids = all_ids[n_train + n_val:]

print(f"Total IDs: {n} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# ============================================================
# BERT backbone (not used for training here, but kept)
# ============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()


def get_text_embedding(text: str):
    inputs = tokenizer(
        text[:512],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    return token_embeddings, attention_mask

# ============================================================
# CNN backbone for visual heuristics (not used in MLP training)
# ============================================================

cnn = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
num_features_cnn = cnn.classifier.in_features
cnn.classifier = nn.Identity()
for param in cnn.parameters():
    param.requires_grad = True
for param in cnn.features.denseblock4.parameters():
    param.requires_grad = True
for param in cnn.features.norm5.parameters():
    param.requires_grad = True
cnn = cnn.to(device)
cnn.train()

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ============================================================
# HuggingFace sentiment pipelines (for text SAW)
# ============================================================

try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
    )
except Exception:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

try:
    fineweb_pipeline = pipeline(
        "sentiment-analysis",
        model="michellejieli/NLTK_based-twitter-sentiment-analysis",
    )
except Exception:
    fineweb_pipeline = None

# ============================================================
# Lexicons / helpers
# ============================================================

positive_words = {
    "good", "great", "excellent", "amazing", "love", "best", "wonderful",
    "fantastic", "beautiful", "awesome", "perfect", "happy", "joy", "glad",
    "brilliant", "outstanding", "superb", "terrific", "lovely", "exceptional",
    "gorgeous", "divine", "marvelous", "splendid", "magnificent", "impressive",
    "delightful", "enjoyable",
}

negative_words = {
    "bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting",
    "ugly", "poor", "sad", "angry", "disappointed", "pathetic", "useless",
    "dumb", "stupid", "sucks", "gross", "weak", "fail", "failure",
    "disappointing", "annoying", "frustrating", "mediocre", "atrocious",
    "dreadful", "abysmal", "repulsive", "appalling",
}

negation_words = {
    "no", "not", "neither", "nor", "never", "nobody", "nothing", "n't",
    "couldn't", "shouldn't", "wouldn't", "don't", "doesn't", "didn't",
    "won't", "can't", "isn't",
}


def split_sentences(text: str):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]

# ============================================================
# SAW utilities
# ============================================================

def saw(decision_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = decision_matrix.max(axis=0)
    denom[denom == 0] = 1.0
    norm_matrix = decision_matrix / denom
    return norm_matrix @ weights


def saw_normalize(D: np.ndarray) -> np.ndarray:
    maxs = D.max(axis=0)
    mins = D.min(axis=0)
    diff = maxs - mins
    diff[diff == 0] = 1.0
    return (D - mins) / diff


def final_text_label(pos: float, neg: float, neu: float) -> str:
    scores = np.array([pos, neg, neu])
    dominant = np.argmax(scores)
    sorted_scores = np.sort(scores)[::-1]
    if sorted_scores[0] - sorted_scores[1] > 0.10:
        return ["positive", "negative", "neutral"][dominant]
    return ["positive", "negative", "neutral"][dominant]


def text_saw_classifier(word_scores: np.ndarray, sent_scores: np.ndarray) -> str:
    D = np.vstack([word_scores, sent_scores])
    R = saw_normalize(D)
    W = np.array([0.5, 0.5])
    saw_scores = R.T @ W
    pos, neg, neu = saw_scores
    return final_text_label(pos, neg, neu)

# ============================================================
# Text / image / audio scoring (heuristics for SAW)
# ============================================================

def text_scores(text: str):
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        result = sentiment_pipeline(text[:512])[0]
        label = result["label"].lower()
        confidence = result["score"]

        if "positive" in label or label == "pos":
            scores["positive"] += confidence * 0.4
            scores["negative"] += (1 - confidence) * 0.12
            scores["neutral"] += (1 - confidence) * 0.28
        elif "negative" in label or label == "neg":
            scores["negative"] += confidence * 0.4
            scores["positive"] += (1 - confidence) * 0.12
            scores["neutral"] += (1 - confidence) * 0.28
        else:
            scores["neutral"] += confidence * 0.4
            scores["positive"] += (1 - confidence) * 0.3
            scores["negative"] += (1 - confidence) * 0.3

        if fineweb_pipeline is not None:
            try:
                result2 = fineweb_pipeline(text[:512])[0]
                label2 = result2["label"].lower()
                confidence2 = result2["score"]
                if "positive" in label2:
                    scores["positive"] += confidence2 * 0.15
                elif "negative" in label2:
                    scores["negative"] += confidence2 * 0.15
                else:
                    scores["neutral"] += confidence2 * 0.15
            except Exception:
                pass

        sentences = split_sentences(text)
        for sentence in sentences:
            sent_lower = sentence.lower()
            tokens = sent_lower.split()
            has_negation = any(neg in tokens for neg in negation_words)
            pos_count = sum(1 for w in positive_words if w in sent_lower)
            neg_count = sum(1 for w in negative_words if w in sent_lower)
            if has_negation:
                if pos_count > 0:
                    scores["negative"] += pos_count * 0.08
                if neg_count > 0:
                    scores["positive"] += neg_count * 0.08
            else:
                if pos_count > neg_count:
                    scores["positive"] += (pos_count - neg_count) * 0.08
                elif neg_count > pos_count:
                    scores["negative"] += (neg_count - pos_count) * 0.08

        exclamation_count = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if exclamation_count > 0:
            scores["positive"] += exclamation_count * 0.05
        if caps_ratio > 0.25:
            scores["positive"] += 0.1

        total = sum(scores.values())
        if total > 0:
            word_probs = np.array(
                [
                    scores["positive"] / total,
                    scores["negative"] / total,
                    scores["neutral"] / total,
                ]
            )
        else:
            word_probs = np.array([0.33, 0.33, 0.34])

        sent_probs = word_probs * 0.9 + np.array([0.33, 0.33, 0.34]) * 0.1
        sent_probs = sent_probs / sent_probs.sum()

    except Exception as e:
        print(f"Error in sentiment analysis (text_scores): {e}")
        word_probs = np.array([0.33, 0.33, 0.34])
        sent_probs = np.array([0.33, 0.33, 0.34])

    return word_probs, sent_probs


def image_scores_from_visual(visual_seq: np.ndarray):
    if visual_seq.ndim != 2:
        return (
            np.array([0.33, 0.33, 0.34]),
            np.array([0.33, 0.33, 0.34]),
            np.array([0.33, 0.33, 0.34]),
        )
    mean_vec = visual_seq.mean(axis=0)
    std_vec = visual_seq.std(axis=0)
    energy = float(np.linalg.norm(mean_vec))
    variability = float(np.linalg.norm(std_vec))
    pos_score = max(0.0, energy * 0.4 + variability * 0.2)
    neg_score = max(0.0, (1.0 / (1.0 + energy)) * 0.6 + variability * 0.1)
    neu_score = max(0.0, 1.0 - (abs(energy - variability) * 0.1))
    total = pos_score + neg_score + neu_score
    if total > 0:
        adjusted_scores = np.array(
            [pos_score / total, neg_score / total, neu_score / total]
        )
    else:
        adjusted_scores = np.array([0.33, 0.33, 0.34])
    obj_scores = adjusted_scores * 0.9 + 0.05
    scene_scores = adjusted_scores * 1.05
    scene_scores = scene_scores / scene_scores.sum()
    return adjusted_scores, obj_scores, scene_scores


def audio_scores(audio_seq: np.ndarray):
    if audio_seq.ndim != 2:
        return np.array([0.33, 0.33, 0.34])
    mean_vec = audio_seq.mean(axis=0)
    std_vec = audio_seq.std(axis=0)
    energy = float(np.linalg.norm(mean_vec))
    dynamics = float(np.linalg.norm(std_vec))
    pos_score = max(0.0, energy * 0.4 + dynamics * 0.2)
    neg_score = max(0.0, (1.0 / (1.0 + energy)) * 0.5 + (1.0 / (1.0 + dynamics)) * 0.2)
    neu_score = max(0.0, 1.0 - abs(energy - dynamics) * 0.1)
    total = pos_score + neg_score + neu_score
    if total > 0:
        probs = np.array(
            [pos_score / total, neg_score / total, neu_score / total]
        )
    else:
        probs = np.array([0.33, 0.33, 0.34])
    return probs

# ============================================================
# NN heads
# ============================================================

## Base
# class TextSentimentMLP(nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=256, num_classes=3, dropout=0.4):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
#         x = self.fc2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
#         logits = self.fc_out(x)
#         return logits

class TextSentimentMLP(nn.Module):
    def __init__(self, input_dim=768, num_classes=3, hidden_dim=(512, 256, 128), dropout=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.feature = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.feature(x)
        return self.fc_out(x)

# class ImageSentimentMLP(nn.Module):
#     def __init__(self, input_dim, num_classes=3, dropout=0.5):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 1024)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.fc_out = nn.Linear(256, num_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
#         x = self.fc2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
#         x = self.fc3(x); x = self.bn3(x); x = F.relu(x); x = self.dropout(x)
#         logits = self.fc_out(x)
#         return logits

class ImageSentimentMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dims=(256, 128), dropout=0.4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.feature = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.feature(x)
        return self.fc_out(x)


# class AudioSentimentMLP(nn.Module):
#     def __init__(self, input_dim, num_classes=3, hidden_dim=128, dropout=0.4):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
#         x = self.fc2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
#         logits = self.fc_out(x)
#         return logits

class AudioSentimentMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=(128, 64), dropout=0.4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.feature = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.feature(x)
        return self.fc_out(x)

# ============================================================
# MCDM methods
# ============================================================

def text_sentiment_tag_saw_only_from_embedding(text_vec_cls: np.ndarray):
    v = text_vec_cls
    energy = float(np.linalg.norm(v))
    pos = energy * 0.4
    neg = (1.0 / (1.0 + energy)) * 0.4
    neu = 1.0
    total = pos + neg + neu
    scores = np.array([pos / total, neg / total, neu / total])
    return LABELS[int(np.argmax(scores))]


METHODS = ["SAW", "TOPSIS", "RAFSI", "TODIM", "MARCOS"]


def minmax_norm(D):
    mn, mx = D.min(axis=0), D.max(axis=0)
    diff = np.where(mx - mn == 0, 1, mx - mn)
    return (D - mn) / diff


def mcd_saw(D, W):
    return minmax_norm(D) @ W


def topsis(D, W):
    R = D / np.sqrt((D ** 2).sum(axis=0))
    V = R * W
    d_pos = np.linalg.norm(V - V.max(axis=0), axis=1)
    d_neg = np.linalg.norm(V - V.min(axis=0), axis=1)
    return d_neg / (d_pos + d_neg + 1e-9)


def rafsi(D, W):
    R = minmax_norm(D)
    ranks = R.argsort(axis=0).argsort(axis=0)
    return (ranks * W).sum(axis=1)


def todim(D, W, theta=1.0):
    R = minmax_norm(D)
    n = R.shape[0]
    score = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = R[i] - R[j]
            score[i] += (
                np.sum(W * np.maximum(diff, 0))
                - np.sum(W * np.maximum(-diff, 0)) / theta
            )
    return score


def marcos(D, W):
    S = minmax_norm(D) @ W
    return S / (S.sum() + 1e-9)


def method_scores(D, W, method):
    if method == "SAW":
        return mcd_saw(D, W)
    if method == "TOPSIS":
        return topsis(D, W)
    if method == "RAFSI":
        return rafsi(D, W)
    if method == "TODIM":
        return todim(D, W)
    if method == "MARCOS":
        return marcos(D, W)
    raise ValueError(f"Unknown method: {method}")

# ============================================================
# Evaluation helper
# ============================================================

def evaluate(y_true, y_pred, name):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(
        f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )
    print(
        f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )
    print(
        f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

# ============================================================
# Original simple feature builder
# ============================================================

def build_split_features(id_list):
    text_emb_list = []
    img_feat_list = []
    audio_feat_list = []
    labels_list = []
    for uid in id_list:
        a = np.asarray(audio_dict[uid], dtype=np.float32)
        v = np.asarray(visual_dict[uid], dtype=np.float32)
        t = np.asarray(text_emb_dict[uid], dtype=np.float32)
        if a.ndim != 2 or v.ndim != 2 or t.ndim != 2:
            continue
        label_s = label_dict[uid]
        try:
            y = encode_label(label_s)
        except ValueError:
            continue
        text_vec = t[0]
        img_vec = v.mean(axis=0)
        audio_vec = a.mean(axis=0)
        text_emb_list.append(text_vec)
        img_feat_list.append(img_vec)
        audio_feat_list.append(audio_vec)
        labels_list.append(y)
    return (
        np.array(text_emb_list, dtype=np.float32),
        np.array(img_feat_list, dtype=np.float32),
        np.array(audio_feat_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int64),
    )

# ============================================================
# Flexible feature builder for ablation
# ============================================================

def build_split_features_for_setup(
    id_list,
    use_text=True,
    use_image=True,
    use_audio=True,
    text_emb_source=None,
):
    text_emb_list = []
    img_feat_list = []
    audio_feat_list = []
    labels_list = []

    text_src = text_emb_dict if text_emb_source is None else text_emb_source

    for uid in id_list:
        if uid not in label_dict:
            continue

        # text
        if use_text:
            if uid not in text_src:
                continue
            t = np.asarray(text_src[uid], dtype=np.float32)
            if t.ndim == 2:
                text_vec = t[0]
            else:
                text_vec = t
        else:
            text_vec = np.zeros(1, dtype=np.float32)

        # image
        if use_image:
            if uid not in visual_dict:
                continue
            v = np.asarray(visual_dict[uid], dtype=np.float32)
            if v.ndim != 2:
                continue
            img_vec = v.mean(axis=0)
        else:
            img_vec = np.zeros(1, dtype=np.float32)

        # audio
        if use_audio:
            if uid not in audio_dict:
                continue
            a = np.asarray(audio_dict[uid], dtype=np.float32)
            if a.ndim != 2:
                continue
            audio_vec = a.mean(axis=0)
        else:
            audio_vec = np.zeros(1, dtype=np.float32)

        label_s = label_dict[uid]
        try:
            y = encode_label(label_s)
        except ValueError:
            continue

        text_emb_list.append(text_vec)
        img_feat_list.append(img_vec)
        audio_feat_list.append(audio_vec)
        labels_list.append(y)

    if not labels_list:
        return None

    return (
        np.array(text_emb_list, dtype=np.float32),
        np.array(img_feat_list, dtype=np.float32),
        np.array(audio_feat_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int64),
    )

# ============================================================
# Training (hyperparameter search) for text/image/audio
# ============================================================

def train_neural_networks_mosi(
    text_embeddings_np,
    image_features_np,
    audio_features_np,
    labels_arr,
    param_grid=None,
    max_trials=5,
):
    if len(labels_arr) == 0:
        print("No training data, skipping training.")
        return None, None, None

    if param_grid is None:
        param_grid = {
            "dropout": [0.3, 0.4, 0.5],
            "lr": [0.0005, 0.001, 0.005, 0.01],
            "batch_size": [32, 64, 128],
            "epochs": [30, 50, 70],
        }

    grid = list(ParameterGrid(param_grid))[:max_trials]

    img_mean = image_features_np.mean(axis=0, keepdims=True)
    img_std = image_features_np.std(axis=0, keepdims=True) + 1e-6
    image_features_np = (image_features_np - img_mean) / img_std

    aud_mean = audio_features_np.mean(axis=0, keepdims=True)
    aud_std = audio_features_np.std(axis=0, keepdims=True) + 1e-6
    audio_features_np = (audio_features_np - aud_mean) / aud_std

    best_val_acc = 0.0
    best_params = None
    best_text_model = None
    best_image_model = None
    best_audio_model = None

    label_counts = Counter(labels_arr)
    total = len(labels_arr)
    class_weights = torch.tensor(
        [total / (3 * label_counts.get(c, 1)) for c in range(3)],
        dtype=torch.float32,
    ).to(device)

    text_dim = text_embeddings_np.shape[1]
    img_dim = image_features_np.shape[1]
    aud_dim = audio_features_np.shape[1]

    for params in grid:
        text_model = TextSentimentMLP(
            input_dim=text_dim,
            hidden_dim=(512, 256, 128),
            num_classes=3,
            dropout=params["dropout"],
        ).to(device)

        image_model = ImageSentimentMLP(
            input_dim=img_dim,
            num_classes=3,
            dropout=params["dropout"],
        ).to(device)

        audio_model = AudioSentimentMLP(
            input_dim=aud_dim,
            num_classes=3,
            hidden_dim=(128, 64),
            dropout=params["dropout"],
        ).to(device)

        text_optimizer = optim.Adam(
            text_model.parameters(), lr=params["lr"], weight_decay=1e-4
        )
        image_optimizer = optim.Adam(
            image_model.parameters(), lr=params["lr"], weight_decay=1e-4
        )
        audio_optimizer = optim.Adam(
            audio_model.parameters(), lr=params["lr"], weight_decay=1e-4
        )

        batch_size = params["batch_size"]
        epochs = params["epochs"]

        text_tensor = torch.from_numpy(text_embeddings_np).float().to(device)
        image_tensor = torch.from_numpy(image_features_np).float().to(device)
        audio_tensor = torch.from_numpy(audio_features_np).float().to(device)
        labels_tensor = torch.from_numpy(labels_arr).long().to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        num_samples = len(labels_arr)
        num_batches = max(1, (num_samples + batch_size - 1) // batch_size)

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        for epoch in range(epochs):
            text_model.train()
            image_model.train()
            audio_model.train()

            perm = torch.randperm(num_samples)
            text_epoch = text_tensor[perm]
            image_epoch = image_tensor[perm]
            audio_epoch = audio_tensor[perm]
            labels_epoch = labels_tensor[perm]

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, num_samples)

                xb_text = text_epoch[start:end]
                xb_image = image_epoch[start:end]
                xb_audio = audio_epoch[start:end]
                yb = labels_epoch[start:end]

                text_optimizer.zero_grad()
                logits_text = text_model(xb_text)
                loss_text = criterion(logits_text, yb)
                loss_text.backward()
                nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
                text_optimizer.step()

                image_optimizer.zero_grad()
                logits_image = image_model(xb_image)
                loss_image = criterion(logits_image, yb)
                loss_image.backward()
                nn.utils.clip_grad_norm_(image_model.parameters(), max_norm=1.0)
                image_optimizer.step()

                audio_optimizer.zero_grad()
                logits_audio = audio_model(xb_audio)
                loss_audio = criterion(logits_audio, yb)
                loss_audio.backward()
                nn.utils.clip_grad_norm_(audio_model.parameters(), max_norm=1.0)
                audio_optimizer.step()

            # Early stopping: use a split of train as pseudo-val
            text_model.eval()
            with torch.no_grad():
                val_logits = text_model(text_tensor)
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                val_acc = (val_preds == labels_arr).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best models
                best_text_model = text_model
                best_image_model = image_model
                best_audio_model = audio_model
                best_params = params
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        text_model.eval()
        image_model.eval()
        audio_model.eval()
        with torch.no_grad():
            logits_text = text_model(text_tensor)
            preds_text = torch.argmax(logits_text, dim=1).cpu().numpy()
            text_acc = (preds_text == labels_arr).mean()

            logits_image = image_model(image_tensor)
            preds_image = torch.argmax(logits_image, dim=1).cpu().numpy()
            image_acc = (preds_image == labels_arr).mean()

            logits_audio = audio_model(audio_tensor)
            preds_audio = torch.argmax(logits_audio, dim=1).cpu().numpy()
            audio_acc = (preds_audio == labels_arr).mean()

        print(
            f"Params: {params} | Text train-Acc: {text_acc:.4f} | "
            f"Image train-Acc: {image_acc:.4f} | Audio train-Acc: {audio_acc:.4f} | Val-Acc: {best_val_acc:.4f}"
        )

    print(f"\nBest Params (val-set): {best_params}")

    best_image_model.img_mean = img_mean
    best_image_model.img_std = img_std
    best_audio_model.aud_mean = aud_mean
    best_audio_model.aud_std = aud_std

    return best_text_model, best_image_model, best_audio_model

# ============================================================
# Run one ablation setup
# ============================================================

def run_setup(
    setup_name,
    use_text=True,
    use_image=True,
    use_audio=True,
    text_emb_source=None,
    param_grid=None,
):
    print("\n" + "=" * 60)
    print(f"SETUP: {setup_name}")
    print("=" * 60)

    train_feats = build_split_features_for_setup(
        train_ids,
        use_text=use_text,
        use_image=use_image,
        use_audio=use_audio,
        text_emb_source=text_emb_source,
    )
    val_feats = build_split_features_for_setup(
        val_ids,
        use_text=use_text,
        use_image=use_image,
        use_audio=use_audio,
        text_emb_source=text_emb_source,
    )
    test_feats = build_split_features_for_setup(
        test_ids,
        use_text=use_text,
        use_image=use_image,
        use_audio=use_audio,
        text_emb_source=text_emb_source,
    )

    if train_feats is None or val_feats is None or test_feats is None:
        print("No data for this setup, skipping.")
        return

    text_train, img_train, aud_train, y_train = train_feats
    text_val, img_val, aud_val, y_val = val_feats
    text_test, img_test, aud_test, y_test = test_feats

    print(f"Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)}")

    text_nn, img_nn, aud_nn = train_neural_networks_mosi(
        text_train, img_train, aud_train, y_train, param_grid=param_grid, max_trials=3
    )

    img_mean = getattr(img_nn, "img_mean", None)
    img_std = getattr(img_nn, "img_std", None)
    aud_mean = getattr(aud_nn, "aud_mean", None)
    aud_std = getattr(aud_nn, "aud_std", None)

    y_true = []
    text_preds_saw = []
    text_preds_nn = []
    img_preds_saw = []
    img_preds_nn = []
    aud_preds_saw = []
    aud_preds_nn = []
    fused_nn_preds = []
    mcdm_multi_preds = {m: [] for m in METHODS}
    hybrid_preds = []
    # Weighted fusion weights (tune as needed)
    fusion_weights = np.array([0.4, 0.3, 0.3])  # text, image, audio

    text_src = text_emb_dict if text_emb_source is None else text_emb_source

    for uid in test_ids:
        if uid not in label_dict:
            continue

        # build per-uid vectors consistent with training dims
        if use_text:
            if uid not in text_src:
                continue
            t_arr = np.asarray(text_src[uid], dtype=np.float32)
            if t_arr.ndim != 2:
                continue
            t_vec = t_arr[0]
        else:
            t_vec = np.zeros(text_train.shape[1], dtype=np.float32)

        if use_image:
            if uid not in visual_dict:
                continue
            v_arr = np.asarray(visual_dict[uid], dtype=np.float32)
            if v_arr.ndim != 2:
                continue
            v_vec = v_arr.mean(axis=0)
        else:
            v_vec = np.zeros(img_train.shape[1], dtype=np.float32)

        if use_audio:
            if uid not in audio_dict:
                continue
            a_arr = np.asarray(audio_dict[uid], dtype=np.float32)
            if a_arr.ndim != 2:
                continue
            a_vec = a_arr.mean(axis=0)
        else:
            a_vec = np.zeros(aud_train.shape[1], dtype=np.float32)

        label_s = label_dict[uid]
        try:
            y = encode_label(label_s)
        except ValueError:
            continue

        # Text SAW-only
        if use_text:
            t_pred_saw = text_sentiment_tag_saw_only_from_embedding(t_vec)
        else:
            t_pred_saw = "neutral"
        text_preds_saw.append(t_pred_saw)

        # Text NN
        tv = torch.from_numpy(t_vec[None, :]).float().to(device)
        with torch.no_grad():
            logits_t = text_nn(tv)
            probs_t = F.softmax(logits_t, dim=1)[0].cpu().numpy()
        text_preds_nn.append(LABELS[int(np.argmax(probs_t))])

        # Image SAW-only
        if use_image:
            g, o, s = image_scores_from_visual(np.tile(v_vec[None, :], (3, 1)))
            D_img = np.array(
                [
                    [g[0], o[0], s[0]],
                    [g[1], o[1], s[1]],
                    [g[2], o[2], s[2]],
                ]
            )
            W_img = np.array([0.6, 0.25, 0.15])
            scores_img_saw = saw(D_img, W_img)
            img_preds_saw.append(LABELS[int(np.argmax(scores_img_saw))])
        else:
            img_preds_saw.append("neutral")

        # Image NN
        if img_mean is not None and img_std is not None:
            img_norm = (v_vec[None, :] - img_mean) / img_std
        else:
            img_norm = v_vec[None, :]
        iv = torch.from_numpy(img_norm).float().to(device)
        with torch.no_grad():
            logits_i = img_nn(iv)
            probs_i = F.softmax(logits_i, dim=1)[0].cpu().numpy()
        img_preds_nn.append(LABELS[int(np.argmax(probs_i))])

        # Audio SAW-only
        if use_audio:
            probs_a_saw = audio_scores(np.tile(a_vec[None, :], (3, 1)))
            aud_preds_saw.append(LABELS[int(np.argmax(probs_a_saw))])
        else:
            aud_preds_saw.append("neutral")

        # Audio NN
        if aud_mean is not None and aud_std is not None:
            aud_norm = (a_vec[None, :] - aud_mean) / aud_std
        else:
            aud_norm = a_vec[None, :]
        av = torch.from_numpy(aud_norm).float().to(device)
        with torch.no_grad():
            logits_a = aud_nn(av)
            probs_a = F.softmax(logits_a, dim=1)[0].cpu().numpy()
        aud_preds_nn.append(LABELS[int(np.argmax(probs_a))])

        # --- Unified Decision Matrix (3 classes x 3 modalities) ---
        decision_matrix = np.zeros((3, 3))
        decision_matrix[:, 0] = probs_t  # text
        decision_matrix[:, 1] = probs_i  # image
        decision_matrix[:, 2] = probs_a  # audio

        weights = np.ones(decision_matrix.shape[1]) / decision_matrix.shape[1]

        # --- MCDM scores per method ---
        m_scores_dict = {}
        for m in METHODS:
            m_scores = method_scores(decision_matrix, weights, m)  # shape (3,)
            m_scores_dict[m] = m_scores
            mcdm_multi_preds[m].append(LABELS[int(np.argmax(m_scores))])

        # --- Learnable NN fusion ---
        class FusionMLP(nn.Module):
            def __init__(self, input_dim=9, hidden_dim=32, dropout=0.5):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.dropout = nn.Dropout(dropout)
                self.fc_out = nn.Linear(hidden_dim, 3)
            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)
                x = F.gelu(x)
                x = self.dropout(x)
                return self.fc_out(x)
        fusion_model = FusionMLP().to(device)
        fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
        fusion_criterion = nn.CrossEntropyLoss()

        # Prepare fusion training data
        fusion_train_X = []
        fusion_train_y = []
        for i in range(len(y_train)):
            tv = torch.from_numpy(text_train[i][None, :]).float().to(device)
            iv = torch.from_numpy(img_train[i][None, :]).float().to(device)
            av = torch.from_numpy(aud_train[i][None, :]).float().to(device)
            with torch.no_grad():
                probs_t = F.softmax(text_nn(tv), dim=1)[0].cpu().numpy()
                probs_i = F.softmax(img_nn(iv), dim=1)[0].cpu().numpy()
                probs_a = F.softmax(aud_nn(av), dim=1)[0].cpu().numpy()
            fusion_train_X.append(np.concatenate([probs_t, probs_i, probs_a]))
            fusion_train_y.append(y_train[i])
        fusion_train_X = torch.tensor(np.array(fusion_train_X), dtype=torch.float32).to(device)
        fusion_train_y = torch.tensor(np.array(fusion_train_y), dtype=torch.long).to(device)
        # Train fusion model
        for epoch in range(30):
            fusion_model.train()
            fusion_optimizer.zero_grad()
            logits = fusion_model(fusion_train_X)
            loss = fusion_criterion(logits, fusion_train_y)
            loss.backward()
            fusion_optimizer.step()

        # Use learnable fusion model for test
        fusion_input = torch.tensor(np.concatenate([probs_t, probs_i, probs_a]), dtype=torch.float32).to(device)
        fusion_model.eval()
        with torch.no_grad():
            fusion_logits = fusion_model(fusion_input[None, :])
            fusion_probs = F.softmax(fusion_logits, dim=1)[0].cpu().numpy()
        fused_nn_preds.append(LABELS[int(np.argmax(fusion_probs))])

        # --- Hybrid NN + MCDM ensemble (NEW) ---
        # normalize MCDM scores per method
        m_norm = {}
        for m, s in m_scores_dict.items():
            s = np.array(s, dtype=np.float32)
            s_min, s_max = s.min(), s.max()
            if s_max - s_min < 1e-9:
                m_norm[m] = np.ones_like(s) / len(s)
            else:
                s_norm = (s - s_min) / (s_max - s_min)
                if s_norm.sum() < 1e-9:
                    s_norm = np.ones_like(s_norm) / len(s_norm)
                else:
                    s_norm = s_norm / s_norm.sum()
                m_norm[m] = s_norm

        selected_methods = ["SAW", "TOPSIS", "MARCOS"]
        mcdm_ensemble = np.zeros(3, dtype=np.float32)
        for m in selected_methods:
            mcdm_ensemble += m_norm[m]
        mcdm_ensemble /= len(selected_methods)

        alpha = 0.6
        hybrid_scores = alpha * fusion_probs + (1.0 - alpha) * mcdm_ensemble
        hybrid_preds.append(LABELS[int(np.argmax(hybrid_scores))])

        y_true.append(LABELS[y])

    print("\nTest samples used:", len(y_true))

    evaluate(y_true, text_preds_saw, f"{setup_name} - TEXT SAW-only")
    evaluate(y_true, text_preds_nn, f"{setup_name} - TEXT NN")
    evaluate(y_true, img_preds_saw, f"{setup_name} - IMAGE SAW-only")
    evaluate(y_true, img_preds_nn, f"{setup_name} - IMAGE NN")
    evaluate(y_true, aud_preds_saw, f"{setup_name} - AUDIO SAW-only")
    evaluate(y_true, aud_preds_nn, f"{setup_name} - AUDIO NN")
    evaluate(y_true, fused_nn_preds, f"{setup_name} - MULTIMODAL NN (avg probs)")

    for m in METHODS:
        evaluate(y_true, mcdm_multi_preds[m], f"{setup_name} - MULTIMODAL {m}")

    # evaluate the new hybrid ensemble once
    evaluate(y_true, hybrid_preds, f"{setup_name} - MULTIMODAL HYBRID (NN + MCDM)")


# ============================================================
# MAIN: run ablation study
# ============================================================

if __name__ == "__main__":
    param_grid = {
        "dropout": [0.4, 0.5, 0.6],
        "lr": [0.0005, 0.001, 0.005, 0.01],
        "batch_size": [32, 64, 128],
        "epochs": [50, 70, 100],
    }

    # Baseline: all three modalities
    run_setup(
        "TEXT+IMAGE+AUDIO (baseline)",
        use_text=True,
        use_image=True,
        use_audio=True,
        text_emb_source=None,
        param_grid=param_grid,
    )

    # # Text-only
    # run_setup(
    #     "TEXT only",
    #     use_text=True,
    #     use_image=False,
    #     use_audio=False,
    #     text_emb_source=None,
    #     param_grid=param_grid,
    # )

    # # Image-only
    # run_setup(
    #     "IMAGE only",
    #     use_text=False,
    #     use_image=True,
    #     use_audio=False,
    #     text_emb_source=None,
    #     param_grid=param_grid,
    # )

    # # Audio-only
    # run_setup(
    #     "AUDIO only",
    #     use_text=False,
    #     use_image=False,
    #     use_audio=True,
    #     text_emb_source=None,
    #     param_grid=param_grid,
    # )

    # # Text+Audio
    # run_setup(
    #     "TEXT+AUDIO",
    #     use_text=True,
    #     use_image=False,
    #     use_audio=True,
    #     text_emb_source=None,
    #     param_grid=param_grid,
    # )

    # # Text+Image
    # run_setup(
    #     "TEXT+IMAGE",
    #     use_text=True,
    #     use_image=True,
    #     use_audio=False,
    #     text_emb_source=None,
    #     param_grid=param_grid,
    # )
