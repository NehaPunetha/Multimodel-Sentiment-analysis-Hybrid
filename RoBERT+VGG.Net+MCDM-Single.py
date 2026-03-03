"""
Enhanced Multimodal sentiment analysis with UNIFIED MCDM Decision Matrix.
Methods: SAW, TOPSIS, RAFSI, TODIM, MARCOS.
Uses MVSA_Single data with:
1. Unified 3×5 decision matrix (Alternatives × Criteria)
2. RoBERTa for text features
3. VGGNet for image features
4. Enhanced hyperparameter tuning
5. Advanced fusion strategies

Decision Matrix Structure:
- Alternatives (3): Positive, Negative, Neutral
- Criteria (5): Word-level Score, Sentence-level Score, Global Score, Object Score, Scene Score
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from transformers import RobertaTokenizer, RobertaModel, pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

import all_MVSA_Single as base


LABELS = ["positive", "negative", "neutral"]
METHODS = ["SAW", "TOPSIS", "RAFSI", "TODIM", "MARCOS"]
CRITERIA = ["Word-level", "Sentence-level", "Global", "Object", "Scene"]

# ============================================================
# DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# RoBERTa MODEL FOR TEXT
# ============================================================
print("Loading RoBERTa for text feature extraction...")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
roberta_model.eval()
print("RoBERTa loaded successfully!")

# RoBERTa sentiment pipeline
print("Loading RoBERTa sentiment analysis pipeline...")
roberta_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1
)
print("RoBERTa sentiment pipeline loaded successfully!")


def get_roberta_embedding(text):
    """Extract RoBERTa embeddings from text."""
    inputs = roberta_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Also get mean of all tokens
        mean_embedding = outputs.last_hidden_state.mean(dim=1)
    
    return cls_embedding, mean_embedding


def get_roberta_sentiment_scores(text):
    """
    Get sentiment scores using RoBERTa pipeline.
    Returns scores for [positive, negative, neutral]
    """
    try:
        result = roberta_sentiment_pipeline(text[:512])[0]  # Limit text length
        label = result['label'].lower()
        score = result['score']
        
        # Initialize scores
        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        # Map labels (cardiffnlp model uses: negative, neutral, positive)
        if 'positive' in label or label == 'pos':
            scores['positive'] = score
            scores['negative'] = (1 - score) / 2
            scores['neutral'] = (1 - score) / 2
        elif 'negative' in label or label == 'neg':
            scores['negative'] = score
            scores['positive'] = (1 - score) / 2
            scores['neutral'] = (1 - score) / 2
        else:  # neutral
            scores['neutral'] = score
            scores['positive'] = (1 - score) / 2
            scores['negative'] = (1 - score) / 2
        
        return np.array([scores['positive'], scores['negative'], scores['neutral']])
    
    except Exception as exc:
        print(f"[Warning] RoBERTa sentiment analysis failed: {exc}")
        return np.array([0.33, 0.33, 0.34])


def get_roberta_word_level_scores(text):
    """
    Compute word-level sentiment scores using RoBERTa.
    Split text into words/phrases and aggregate.
    """
    words = text.split()
    if len(words) == 0:
        return np.array([0.33, 0.33, 0.34])
    
    # For shorter texts, analyze as whole
    if len(words) <= 5:
        return get_roberta_sentiment_scores(text)
    
    # For longer texts, analyze chunks
    chunk_size = 10
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    all_scores = []
    for chunk in chunks:
        if chunk.strip():
            scores = get_roberta_sentiment_scores(chunk)
            all_scores.append(scores)
    
    if all_scores:
        # Average across chunks
        return np.mean(all_scores, axis=0)
    else:
        return np.array([0.33, 0.33, 0.34])


def roberta_text_scores(text):
    """
    Compute both word-level and sentence-level scores using RoBERTa.
    
    Returns:
        word_scores: [pos, neg, neu] from word-level analysis
        sent_scores: [pos, neg, neu] from sentence-level analysis
    """
    # Sentence-level: analyze entire text
    sent_scores = get_roberta_sentiment_scores(text)
    
    # Word-level: analyze in chunks
    word_scores = get_roberta_word_level_scores(text)
    
    return word_scores, sent_scores


# ============================================================
# VGGNet FEATURE EXTRACTOR
# ============================================================
print("Loading VGG16 for image feature extraction...")
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
# Remove the classifier to get features (output: 4096-d from fc7)
vgg_features = nn.Sequential(*list(vgg16.classifier.children())[:-1])
vgg16.classifier = vgg_features
vgg16 = vgg16.to(device)
vgg16.eval()
print("VGG16 loaded successfully!")

# Image transformation for VGG (expects 224x224)
vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_vgg_features(image_path):
    """Extract 4096-d features from VGG16."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = vgg_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16(img_tensor)
    return features


# ============================================================
# ENHANCED IMAGE MODEL ARCHITECTURE (for VGG 4096-d features)
# ============================================================
class EnhancedImageSentimentCNN(nn.Module):
    """
    Enhanced CNN with better architecture for image sentiment using VGG features.
    """
    def __init__(self, input_dim=4096, num_classes=3, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        
        # Multi-scale processing
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Residual connection
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(16)
        
        # Deeper FC layers
        self.fc1 = nn.Linear(256 * 16, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, 4096)
        x = x.unsqueeze(1)  # (B, 1, 4096)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Residual block
        residual = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + residual
        
        x = self.pool(x)  # (B, 256, 16)
        
        # Attention
        pooled = x.mean(dim=2)  # (B, 256)
        att_weights = self.attention(pooled).unsqueeze(2)  # (B, 256, 1)
        x = x * att_weights
        
        x = x.view(x.size(0), -1)  # (B, 256*16)
        
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn_fc3(self.fc3(x))))
        logits = self.fc4(x)
        
        return logits


# ============================================================
# TEXT MODEL ARCHITECTURE (for RoBERTa 768-d features)
# ============================================================
class RoBERTaSentimentMLP(nn.Module):
    """MLP classifier for RoBERTa text embeddings."""
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=3, dropout=0.5):
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


# ============================================================
# IMAGE DATA AUGMENTATION
# ============================================================
def augment_image_features(features, num_augments=3):
    """Create augmented versions of image features for training."""
    augmented = [features]
    for _ in range(num_augments):
        # Add small gaussian noise
        noise = np.random.normal(0, 0.03, features.shape)
        augmented.append(features + noise)
        # Random scaling
        scale = np.random.uniform(0.9, 1.1, (1, features.shape[1]))
        augmented.append(features * scale)
        # Random dropout
        mask = np.random.binomial(1, 0.9, features.shape)
        augmented.append(features * mask)
    return np.vstack(augmented)


# ============================================================
# PREPARE TRAINING DATA WITH RoBERTa + VGG FEATURES
# ============================================================
def prepare_training_data(label_file, data_dir):
    """Extract RoBERTa and VGG features for training."""
    print("Extracting RoBERTa text features and VGG image features...")
    
    text_embeddings = []
    vgg_features_list = []
    labels = []
    
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return [], [], []
    
    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[1:]
    
    count = 0
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        
        idx = parts[0]
        label_str = parts[1].split(",")[0].strip().lower()
        
        if label_str not in ["positive", "negative", "neutral"]:
            continue
        
        label_id = ["positive", "negative", "neutral"].index(label_str)
        
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
        img_jpg = os.path.join(data_dir, f"{idx}.jpg")
        img_png = os.path.join(data_dir, f"{idx}.png")
        if os.path.exists(img_jpg):
            img_path = img_jpg
        elif os.path.exists(img_png):
            img_path = img_png
        
        if img_path is None:
            continue
        
        try:
            # Extract VGG features
            vgg_feat = extract_vgg_features(img_path)
            vgg_feat_np = vgg_feat.cpu().numpy().flatten()
            
            # Extract RoBERTa features
            cls_emb, _ = get_roberta_embedding(text)
            text_vec = cls_emb.cpu().numpy().flatten()
            
            text_embeddings.append(text_vec)
            vgg_features_list.append(vgg_feat_np)
            labels.append(label_id)
            
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count} samples...")
            
        except Exception as exc:
            print(f"Skipping {idx}: {exc}")
            continue
    
    print(f"Extracted features from {len(labels)} samples")
    return text_embeddings, vgg_features_list, labels


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_text_model(text_features, labels, param_grid=None, max_trials=8):
    """Train RoBERTa-based text sentiment model."""
    if len(labels) == 0:
        print("No training data, skipping.")
        return None
    
    if param_grid is None:
        param_grid = {
            "dropout": [0.3, 0.4],
            "lr": [0.001, 0.002],
            "batch_size": [32, 64],
            "epochs": [12],
        }
    
    grid = list(ParameterGrid(param_grid))[:max_trials]
    
    from sklearn.utils import shuffle
    indices = np.arange(len(labels))
    indices = shuffle(indices, random_state=42)
    
    text_features_np = np.array(text_features)[indices]
    labels_arr = np.array(labels)[indices]
    
    # Standardize
    text_mean = text_features_np.mean(axis=0, keepdims=True)
    text_std = text_features_np.std(axis=0, keepdims=True) + 1e-6
    text_features_np = (text_features_np - text_mean) / text_std
    
    # Split
    split_idx = int(0.8 * len(labels_arr))
    train_features = text_features_np[:split_idx]
    val_features = text_features_np[split_idx:]
    train_labels = labels_arr[:split_idx]
    val_labels = labels_arr[split_idx:]
    
    from collections import Counter
    label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (3 * label_counts.get(c, 1)) for c in range(3)],
        dtype=torch.float32,
    ).to(device)
    
    best_val_f1 = 0.0
    best_model = None
    
    print("\nHyperparameter tuning for text model...")
    for idx, params in enumerate(grid, 1):
        print(f"Trial {idx}/{len(grid)}: {params}")
        
        model = RoBERTaSentimentMLP(
            input_dim=768,
            hidden_dim=256,
            num_classes=3,
            dropout=params["dropout"]
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Keep tensors on CPU and move small batches to GPU during training/validation
        train_tensor = torch.from_numpy(train_features).float()
        val_tensor = torch.from_numpy(val_features).float()
        train_labels_tensor = torch.from_numpy(train_labels).long()
        val_labels_tensor = torch.from_numpy(val_labels).long()
        
        batch_size = params["batch_size"]
        epochs = params["epochs"]
        num_batches = max(1, len(train_labels) // batch_size)
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(train_labels))
            train_tensor_shuffled = train_tensor[perm]
            train_labels_shuffled = train_labels_tensor[perm]

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, len(train_labels))

                xb = train_tensor_shuffled[start:end].to(device)
                yb = train_labels_shuffled[start:end].to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation
            model.eval()
            # Run validation in batches to avoid OOM
            val_logits = predict_logits_in_batches(model, val_tensor, batch_size=batch_size)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            
            scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
    
    print(f"Best text model F1: {best_val_f1:.4f}")
    
    # Store normalization params
    best_model.text_mean = text_mean
    best_model.text_std = text_std
    
    return best_model


def train_image_model(image_features, labels, param_grid=None, max_trials=15, use_augmentation=True):
    """Train VGG-based image sentiment model."""
    if len(labels) == 0:
        print("No training data, skipping.")
        return None
    
    if param_grid is None:
        param_grid = {
            "dropout": [0.3, 0.4, 0.5],
            "lr": [0.0005, 0.001, 0.002],
            "batch_size": [16, 32, 64],
            "epochs": [15, 20],
            "weight_decay": [1e-4, 1e-5],
            "optimizer": ["adam", "adamw"],
        }
    
    grid = list(ParameterGrid(param_grid))[:max_trials]
    
    from sklearn.utils import shuffle
    indices = np.arange(len(labels))
    indices = shuffle(indices, random_state=42)
    
    image_features_np = np.array(image_features)[indices]
    labels_arr = np.array(labels)[indices]
    
    if use_augmentation:
        # Avoid exploding dataset size which can lead to OOM. Only augment when
        # the base dataset is reasonably small.
        if len(image_features_np) <= 2000:
            print("Applying data augmentation...")
            aug_features = []
            aug_labels = []
            for feat, label in zip(image_features_np, labels_arr):
                augmented = augment_image_features(feat.reshape(1, -1), num_augments=3)
                aug_features.append(augmented)
                aug_labels.extend([label] * len(augmented))
            image_features_np = np.vstack(aug_features)
            labels_arr = np.array(aug_labels)
            print(f"Augmented dataset size: {len(labels_arr)}")
        else:
            print("Skipping augmentation: dataset too large (avoids OOM)")
    
    # Standardize
    img_mean = image_features_np.mean(axis=0, keepdims=True)
    img_std = image_features_np.std(axis=0, keepdims=True) + 1e-6
    image_features_np = (image_features_np - img_mean) / img_std
    
    # Split
    split_idx = int(0.8 * len(labels_arr))
    train_features = image_features_np[:split_idx]
    val_features = image_features_np[split_idx:]
    train_labels = labels_arr[:split_idx]
    val_labels = labels_arr[split_idx:]
    
    from collections import Counter
    label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (3 * label_counts.get(c, 1)) for c in range(3)],
        dtype=torch.float32,
    ).to(device)
    
    best_val_f1 = 0.0
    best_models = []
    print("\nHyperparameter tuning for image model (ensemble + early stopping)...")
    for idx, params in enumerate(grid, 1):
        print(f"Trial {idx}/{len(grid)}: {params}")
        model = EnhancedImageSentimentCNN(
            input_dim=4096,
            num_classes=3,
            dropout=params["dropout"]
        ).to(device)
        if params["optimizer"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Keep tensors on CPU; move only batches to GPU during training/validation
        train_tensor = torch.from_numpy(train_features).float()
        val_tensor = torch.from_numpy(val_features).float()
        train_labels_tensor = torch.from_numpy(train_labels).long()
        val_labels_tensor = torch.from_numpy(val_labels).long()
        batch_size = params["batch_size"]
        epochs = params["epochs"]
        num_batches = max(1, len(train_labels) // batch_size)
        best_epoch_val_f1 = 0.0
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(train_labels))
            train_tensor_shuffled = train_tensor[perm]
            train_labels_shuffled = train_labels_tensor[perm]
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, len(train_labels))
                xb = train_tensor_shuffled[start:end].to(device)
                yb = train_labels_shuffled[start:end].to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            # Validation
            model.eval()
            # Run validation in batches to avoid large GPU allocation
            val_logits = predict_logits_in_batches(model, val_tensor, batch_size=batch_size)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            scheduler.step(val_f1)
            if val_f1 > best_epoch_val_f1:
                best_epoch_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 3:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        best_models.append((best_epoch_val_f1, model))
        if best_epoch_val_f1 > best_val_f1:
            best_val_f1 = best_epoch_val_f1
    # Ensemble: average predictions from top K models
    best_models = sorted(best_models, key=lambda x: -x[0])[:3]
    print(f"Best image model F1: {best_val_f1:.4f}")
    # Store normalization params in all models
    for _, m in best_models:
        m.img_mean = img_mean
        m.img_std = img_std
    # Return ensemble (list of models)
    return [m for _, m in best_models]


# ============================================================
# BUILD UNIFIED DECISION MATRIX (3×5)
# ============================================================
def build_decision_matrix(text, image_path):
    """
    Build unified 3×5 decision matrix using RoBERTa for text and VGG for images.
    
    Matrix structure:
    Rows (Alternatives): Positive, Negative, Neutral
    Columns (Criteria): Word-level, Sentence-level, Global, Object, Scene
    """
    # Initialize decision matrix
    D = np.zeros((3, 5))
    
    # TEXT FEATURES using RoBERTa
    word_scores, sent_scores = roberta_text_scores(text)
    D[:, 0] = word_scores  # Column 0: Word-level scores
    D[:, 1] = sent_scores  # Column 1: Sentence-level scores
    
    # IMAGE FEATURES
    if image_path is not None:
        try:
            global_scores, object_scores, scene_scores = base.image_scores(image_path)
            D[:, 2] = global_scores  # Column 2: Global scores
            D[:, 3] = object_scores  # Column 3: Object scores
            D[:, 4] = scene_scores   # Column 4: Scene scores
        except Exception as exc:
            print(f"[Warning] Image processing failed: {exc}")
            D[:, 2] = np.array([0.33, 0.33, 0.34])
            D[:, 3] = np.array([0.33, 0.33, 0.34])
            D[:, 4] = np.array([0.33, 0.33, 0.34])
    else:
        D[:, 2] = np.array([0.33, 0.33, 0.34])
        D[:, 3] = np.array([0.33, 0.33, 0.34])
        D[:, 4] = np.array([0.33, 0.33, 0.34])
    
    return D


def print_decision_matrix(D, sample_id=None):
    """Pretty print the decision matrix."""
    if sample_id:
        print(f"\nDecision Matrix for Sample {sample_id}:")
    else:
        print("\nDecision Matrix:")
    print("=" * 90)
    print(f"{'Alternative':<12} | {'Word-lvl':<12} | {'Sent-lvl':<12} | {'Global':<12} | {'Object':<12} | {'Scene':<12}")
    print("-" * 90)
    for i, label in enumerate(LABELS):
        print(f"{label.capitalize():<12} | {D[i,0]:<12.4f} | {D[i,1]:<12.4f} | {D[i,2]:<12.4f} | {D[i,3]:<12.4f} | {D[i,4]:<12.4f}")
    print("=" * 90)


# ============================================================
# MCDM METHODS
# ============================================================
def minmax_norm(D):
    """Min-Max normalization."""
    mn, mx = D.min(axis=0), D.max(axis=0)
    diff = np.where(mx - mn == 0, 1, mx - mn)
    return (D - mn) / diff


def saw(D, W):
    """SAW method."""
    D_norm = minmax_norm(D)
    return D_norm @ W


def topsis(D, W):
    """TOPSIS method."""
    R = D / np.sqrt((D ** 2).sum(axis=0) + 1e-9)
    V = R * W
    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)
    d_pos = np.linalg.norm(V - ideal_best, axis=1)
    d_neg = np.linalg.norm(V - ideal_worst, axis=1)
    return d_neg / (d_pos + d_neg + 1e-9)


def rafsi(D, W):
    """RAFSI method."""
    R = minmax_norm(D)
    ranks = R.argsort(axis=0).argsort(axis=0)
    return (ranks * W).sum(axis=1)


def todim(D, W, theta=1.0):
    """TODIM method."""
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
    """MARCOS method."""
    D_norm = minmax_norm(D)
    S = D_norm @ W
    return S / (S.sum() + 1e-9)


def method_scores(D, W, method):
    """Apply MCDM method."""
    if method == "SAW":
        return saw(D, W)
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
# ENHANCED FUSION STRATEGIES
# ============================================================
def adaptive_fuse(mcdm_scores, nn_probs, high=0.70, mid=0.55,
                  high_w=0.80, mid_w=0.70, low_w=0.55):
    """Adaptive fusion."""
    nn_conf = nn_probs.max()
    if nn_conf > high:
        nn_weight = high_w
    elif nn_conf > mid:
        nn_weight = mid_w
    else:
        nn_weight = low_w
    return mcdm_scores * (1.0 - nn_weight) + nn_probs * nn_weight


def ensemble_fusion(mcdm_scores, nn_probs, temperature=1.5):
    """Temperature-scaled ensemble fusion."""
    mcdm_exp = np.exp(mcdm_scores / temperature)
    nn_exp = np.exp(nn_probs / temperature)
    
    mcdm_soft = mcdm_exp / (mcdm_exp.sum() + 1e-9)
    nn_soft = nn_exp / (nn_exp.sum() + 1e-9)
    
    return 0.4 * mcdm_soft + 0.6 * nn_soft


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def text_nn_probs(text, model):
    """Get text sentiment probabilities using RoBERTa + NN."""
    cls_emb, _ = get_roberta_embedding(text)
    text_vec = cls_emb.to(device)

    if hasattr(model, "text_mean") and model.text_mean is not None:
        vec = text_vec.cpu().numpy()
        vec = (vec - model.text_mean) / model.text_std
        text_vec = torch.from_numpy(vec).float().to(device)

    with torch.no_grad():
        logits = model(text_vec)
        return F.softmax(logits, dim=1)[0].cpu().numpy()


def image_nn_probs(image_path, model):
    """Get image sentiment probabilities using VGG + NN."""
    img_feat = extract_vgg_features(image_path)

    # Support ensembles: if `model` is a list of models, average their probs
    if isinstance(model, list):
        probs_list = []
        for m in model:
            feat = img_feat.cpu().numpy()
            if hasattr(m, "img_mean") and m.img_mean is not None:
                feat = (feat - m.img_mean) / m.img_std
            inp = torch.from_numpy(feat).float().to(device)
            with torch.no_grad():
                logits = m(inp)
                probs_list.append(F.softmax(logits, dim=1)[0].cpu().numpy())
        if probs_list:
            return np.mean(probs_list, axis=0)
        else:
            return np.array([0.33, 0.33, 0.34])

    # Single model case
    feat = img_feat.cpu().numpy()
    if hasattr(model, "img_mean") and model.img_mean is not None:
        feat = (feat - model.img_mean) / model.img_std
    img_feat = torch.from_numpy(feat).float().to(device)
    with torch.no_grad():
        logits = model(img_feat)
        return F.softmax(logits, dim=1)[0].cpu().numpy()


def predict_logits_in_batches(model, input_tensor, batch_size=64):
    """Run model on `input_tensor` in small batches and return concatenated logits (CPU tensor).

    `input_tensor` should be a torch.Tensor (CPU or device); batches are moved to `device`.
    """
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
    else:
        return torch.empty((0, model.fc4.out_features if hasattr(model, 'fc4') else 0))


def multimodal_nn_probs(text, image_path, text_model, image_model, text_weight=0.5):
    """Combine text and image NN predictions."""
    text_probs = text_nn_probs(text, text_model)
    
    if image_path is not None:
        try:
            image_probs = image_nn_probs(image_path, image_model)
            return text_weight * text_probs + (1 - text_weight) * image_probs
        except Exception as exc:
            print(f"[Warning] Image NN failed: {exc}")
            return text_probs
    else:
        return text_probs


# ============================================================
# EVALUATION
# ============================================================
def evaluate(y_true, y_pred, name):
    """Evaluate and print metrics."""
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%")
    print(f"F1-score : {f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 90)
    print("UNIFIED MCDM SENTIMENT ANALYSIS WITH RoBERTa + VGG16")
    print("Decision Matrix: 3 Alternatives × 5 Criteria")
    print("=" * 90)
    
    print("\n" + "=" * 90)
    print("PREPARING TRAINING DATA")
    print("=" * 90)

    text_emb_train, vgg_feat_train, labels_train = prepare_training_data(
        base.LABEL_FILE, base.DATA_DIR
    )

    print(f"Collected {len(labels_train)} training samples.")

    if len(labels_train) > 10:
        print("\n" + "=" * 90)
        print("TRAINING NEURAL NETWORKS")
        print("=" * 90)

        # Train RoBERTa text model
        print("\nTraining RoBERTa-based text model...")
        text_model = train_text_model(text_emb_train, labels_train)
        
        # Train VGG image model
        print("\n" + "=" * 90)
        print("Training VGG-based image model...")
        print("=" * 90)
        image_model = train_image_model(vgg_feat_train, labels_train, max_trials=15, use_augmentation=True)
    else:
        print("Not enough training samples. Using randomly initialized networks.")
        text_model = RoBERTaSentimentMLP().to(device)
        image_model = EnhancedImageSentimentCNN(input_dim=4096).to(device)

    # Define criteria weights
    W = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    
    print(f"\n{'='*90}")
    print(f"MCDM Criteria Weights:")
    print(f"  Word-level (RoBERTa): {W[0]:.2f}")
    print(f"  Sentence-level (RoBERTa): {W[1]:.2f}")
    print(f"  Global (VGG): {W[2]:.2f}")
    print(f"  Object (VGG): {W[3]:.2f}")
    print(f"  Scene (VGG): {W[4]:.2f}")
    print(f"{'='*90}\n")

    # Initialize prediction storage
    mcdm_preds = {m: [] for m in METHODS}
    mcdm_nn_adaptive = {m: [] for m in METHODS}
    mcdm_nn_ensemble = {m: [] for m in METHODS}
    nn_only_preds = []
    true_labels = []

    print("\n" + "=" * 90)
    print("RUNNING UNIFIED MCDM EVALUATION")
    print("=" * 90)

    if not os.path.exists(base.LABEL_FILE):
        print(f"Cannot evaluate, label file not found: {base.LABEL_FILE}")
        raise SystemExit(0)

    with open(base.LABEL_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[1:]

    processed_count = 0
    show_sample_matrices = 3
    
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        idx = parts[0]
        labels_split = parts[1].split(",")
        t_label = labels_split[0].strip().lower()

        text_file = os.path.join(base.DATA_DIR, f"{idx}.txt")
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
        img_jpg = os.path.join(base.DATA_DIR, f"{idx}.jpg")
        img_png = os.path.join(base.DATA_DIR, f"{idx}.png")
        if os.path.exists(img_jpg):
            img_path = img_jpg
        elif os.path.exists(img_png):
            img_path = img_png

        # BUILD UNIFIED DECISION MATRIX
        D = build_decision_matrix(text, img_path)
        
        if processed_count < show_sample_matrices:
            print_decision_matrix(D, sample_id=idx)

        # Apply MCDM methods
        for method in METHODS:
            mcdm_score = method_scores(D, W, method)
            mcdm_preds[method].append(LABELS[int(np.argmax(mcdm_score))])
            
            try:
                nn_probs = multimodal_nn_probs(text, img_path, text_model, image_model)
                
                # Adaptive fusion
                fused_adaptive = adaptive_fuse(mcdm_score, nn_probs, 
                                              high=0.70, mid=0.55,
                                              high_w=0.85, mid_w=0.75, low_w=0.60)
                mcdm_nn_adaptive[method].append(LABELS[int(np.argmax(fused_adaptive))])
                
                # Ensemble fusion
                fused_ensemble = ensemble_fusion(mcdm_score, nn_probs)
                mcdm_nn_ensemble[method].append(LABELS[int(np.argmax(fused_ensemble))])
                
                if len(nn_only_preds) <= processed_count:
                    nn_only_preds.append(LABELS[int(np.argmax(nn_probs))])
                    
            except Exception as exc:
                print(f"[Warning] NN prediction failed for {idx}: {exc}")
                mcdm_nn_adaptive[method].append(LABELS[int(np.argmax(mcdm_score))])
                mcdm_nn_ensemble[method].append(LABELS[int(np.argmax(mcdm_score))])
                if len(nn_only_preds) <= processed_count:
                    nn_only_preds.append(LABELS[int(np.argmax(mcdm_score))])

        true_labels.append(t_label)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} samples...")

    print("\n" + "=" * 90)
    print(f"EVALUATION RESULTS (Total: {processed_count} samples)")
    print("=" * 90)

    # Evaluate pure MCDM
    print("\n" + "=" * 90)
    print("PURE MCDM METHODS (RoBERTa + VGG)")
    print("=" * 90)
    for method in METHODS:
        evaluate(true_labels, mcdm_preds[method], f"{method}")

    # Evaluate MCDM + NN Adaptive
    print("\n" + "=" * 90)
    print("MCDM + NN ADAPTIVE FUSION")
    print("=" * 90)
    for method in METHODS:
        evaluate(true_labels, mcdm_nn_adaptive[method], f"{method} + NN (Adaptive)")

    # Evaluate MCDM + NN Ensemble
    print("\n" + "=" * 90)
    print("MCDM + NN ENSEMBLE FUSION")
    print("=" * 90)
    for method in METHODS:
        evaluate(true_labels, mcdm_nn_ensemble[method], f"{method} + NN (Ensemble)")

    # Evaluate NN only
    print("\n" + "=" * 90)
    print("BASELINE: NEURAL NETWORK ONLY")
    print("=" * 90)
    evaluate(true_labels, nn_only_preds, "RoBERTa + VGG NN")

    print("\n" + "=" * 90)
    print("✓ RoBERTa + VGG MCDM analysis completed successfully!")
    print("=" * 90)