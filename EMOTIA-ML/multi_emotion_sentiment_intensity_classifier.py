"""
multi-task single-emotion classification (emocja + intensywność)
@author: Julia Wieczorek
Multi-task classifier - IMPROVED VERSION
New improvements:
- Fixed duplicate loss_fns bug
- Added learning rate warmup with linear decay
- Integrated Focal Loss for rare emotions
- Added gradient accumulation option
- Better optimizer configuration
"""

import time
import os
import json
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path

import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, logging as hf_logging
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

hf_logging.set_verbosity_error()

# -------------------------
# Config
# -------------------------
DEFAULT_CONFIG = {
    "transformer_model": "bert-base-uncased",
    "bert_model": "soft-sharing",
    "max_len": 128, #128,
    "batch_size":16, #16,
    "epochs": 6, #3, #do testow 6,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "dropout": 0.4, #0.3
    "lstm_hidden_dim": 128,
    "lstm_layers": 1,
    "bidirectional": True,
    "seed": 42,
    "output_dir": "./outputs_multitask",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "w_sentiment": 1.0,
    "w_emotion": 1.0,
    "w_intensity": 0.7,
    "early_stopping_patience": 2, #do testow3,
    # NEW PARAMS
    "use_focal_loss": True,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "warmup_ratio": 0.2,  #0.1 -> 10% kroków to warmup
    "gradient_accumulation_steps": 1,  # Zwiększ do 4 jeśli mało RAM
}

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_binary(val, emotion_name=None):
    """Robust bool parsing for emotion columns"""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return int(val) != 0
    s = str(val).strip().lower()
    if s in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    if emotion_name and s == emotion_name.lower():
        return 1
    return 0

def parse_intensity(val):
    """Convert intensity (1/2/3) to (0/1/2)"""
    if pd.isna(val):
        return 0
    try:
        v = int(float(val))
        if v < 1:
            return 0
        if v > 3:
            return 2
        return v - 1
    except:
        return 0

# -------------------------
# CSV VALIDATION
# -------------------------
def validate_csv_format(df):
    """Validate CSV has required format"""
    errors = []

    # Check text column
    text_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content', 'augmented'])]
    if not text_cols:
        errors.append("No text column found (expected: 'utterance', 'text', 'message', 'augmented' or 'content')")

    # Check sentiment
    if 'sentiment' not in df.columns:
        errors.append("Missing 'sentiment' column")

    # Check emotion/intensity pairs
    emotion_cols = [c for c in df.columns if c.startswith("emotion__")]
    if not emotion_cols:
        errors.append("No emotion columns found (expected format: emotion__<name>)")

    for ecol in emotion_cols:
        emotion_name = ecol.split("emotion__", 1)[1]
        intensity_col = f"intensity__{emotion_name}"
        if intensity_col not in df.columns:
            errors.append(f"Missing intensity column: {intensity_col}")

    if errors:
        raise ValueError("CSV format validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"CSV validation passed:")
    print(f"   - Text column: {text_cols[0]}")
    print(f"   - {len(emotion_cols)} emotion types: {[c.split('__')[1] for c in emotion_cols]}")
    return text_cols[0]

# -------------------------
# Data loading
# -------------------------
def load_multitask_data(csv_path):
    """Load and parse multi-task CSV data"""
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    df.columns = df.columns.str.strip()

    # Validate format and get text column name
    text_col = validate_csv_format(df)

    # Get emotion columns
    emotion_cols = [c for c in df.columns if c.startswith("emotion__")]
    emotion_names = [c.split("emotion__", 1)[1] for c in emotion_cols]

    # Enhanced sentiment mapping
    sentiment_map = {
        "negative": 0, "neg": 0, "0": 0, 0: 0,
        "neutral": 1, "neu": 1, "1": 1, 1: 1,
        "positive": 2, "pos": 2, "2": 2, 2: 2
    }

    texts, sentiments = [], []
    emotion_matrix = []
    intensity_matrix = []

    skipped = 0

    for idx, row in df.iterrows():
        text_val = str(row[text_col]).strip()
        if (not text_val or len(text_val) < 5) and 'original' in df.columns:
            fallback_val = str(row.get('original', '')).strip()
            if len(fallback_val) >= 5:
                text_val = fallback_val

        if not text_val or len(text_val) < 5:
            skipped += 1
            continue

        raw_sent = row.get('sentiment', 'neutral')
        sentiment_label = sentiment_map.get(str(raw_sent).strip().lower(), None)

        if sentiment_label is None:
            try:
                sentiment_label = int(float(raw_sent))
                sentiment_label = max(0, min(2, sentiment_label))
            except:
                print(f"Row {idx}: Invalid sentiment '{raw_sent}', defaulting to neutral")
                sentiment_label = 1

        # Parse emotions and intensities
        em_row = []
        int_row = []

        for ename, ecol in zip(emotion_names, emotion_cols):
            val = row.get(ecol, np.nan)

            if pd.isna(val):
                em_row.append(0)
                int_row.append(0)
            else:
                b = to_binary(val, emotion_name=ename)
                em_row.append(b)

                icol = f"intensity__{ename}"
                ival = row.get(icol, np.nan)
                int_row.append(parse_intensity(ival))

        texts.append(text_val)
        sentiments.append(sentiment_label)
        emotion_matrix.append(em_row)
        intensity_matrix.append(int_row)

    if skipped > 0:
        print(f"Skipped {skipped} rows with empty/invalid text")

    emotion_matrix = np.array(emotion_matrix, dtype=np.int64)
    intensity_matrix = np.array(intensity_matrix, dtype=np.int64)

    print(f"Loaded {len(texts)} valid samples")
    print(f"   Sentiment distribution: {np.bincount(sentiments)}")
    print(f"   Emotion presence (any): {emotion_matrix.sum(axis=1).mean():.2f} avg per sample")

    # -------------------------
    # Statystyka współwystępowania emocji
    # -------------------------
    emotion_counts = emotion_matrix.sum(axis=1)

    single_label_ratio = np.mean(emotion_counts <= 1)
    multi_label_mode = single_label_ratio < 0.95  # jeśli >5% ma wiele emocji → multi-label

    mode_type = "multi-label" if multi_label_mode else "single-emotion"
    print(f"\nDetected emotion labeling mode: {mode_type.upper()}")


    one_emotion = np.sum(emotion_counts == 1)
    two_emotions = np.sum(emotion_counts == 2)
    multi_emotions = np.sum(emotion_counts >= 3)

    print("\nEmotion co-occurrence stats:")
    print(f"   1 emotion present     : {one_emotion} samples ({100*one_emotion/len(emotion_counts):.1f}%)")
    print(f"   2 emotions present    : {two_emotions} samples ({100*two_emotions/len(emotion_counts):.1f}%)")
    print(f"   3+ emotions present   : {multi_emotions} samples ({100*multi_emotions/len(emotion_counts):.1f}%)")

    if intensity_matrix.size > 0:
        avg_intensity_per_emotion = intensity_matrix.mean(axis=0)
        avg_intensity_all = intensity_matrix[intensity_matrix > 0].mean() if (intensity_matrix > 0).any() else 0
        print("\nAverage emotion intensity (0=low, 1=medium, 2=high):")
        for ename, avg_val in zip(emotion_names, avg_intensity_per_emotion):
            print(f"   {ename:15s}: {avg_val:.2f}")
        print(f"   Overall intensity average: {avg_intensity_all:.2f}")

    return texts, np.array(sentiments, dtype=np.int64), emotion_matrix, intensity_matrix, emotion_names, mode_type

class MultiTaskDataset(Dataset):
    def __init__(self, texts, sentiment_labels, emotion_labels, intensity_labels, tokenizer, max_len):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.emotion_labels = emotion_labels
        self.intensity_labels = intensity_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['sentiment'] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        item['emotions'] = torch.tensor(self.emotion_labels[idx], dtype=torch.float)
        item['intensities'] = torch.tensor(self.intensity_labels[idx], dtype=torch.long)
        return item

class MultiTaskBERTLSTM(nn.Module):
    def __init__(self, bert_model, num_emotions, lstm_hidden=128, lstm_layers=1, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.shared_bn = nn.BatchNorm1d(lstm_output_dim // 2)

        # Heads
        self.sentiment_head = nn.Linear(lstm_output_dim // 2, 3)
        self.emotion_head = nn.Linear(lstm_output_dim // 2, num_emotions)
        self.intensity_head = nn.Linear(lstm_output_dim // 2, num_emotions * 3)

        nn.init.xavier_uniform_(self.shared_fc.weight)
        nn.init.constant_(self.shared_fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state

        lstm_out, (h_n, _) = self.lstm(sequence_output)

        if self.lstm.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat((h_forward, h_backward), dim=1)
        else:
            h = h_n[-1]

        x = self.dropout(h)
        x = self.shared_fc(x)
        x = self.shared_bn(x)
        x = torch.relu(x)
        x = self.dropout(x)

        sentiment_logits = self.sentiment_head(x)
        emotion_logits = self.emotion_head(x)
        intensity_logits = self.intensity_head(x)

        return sentiment_logits, emotion_logits, intensity_logits

class MultiTaskBERT(nn.Module):
    def __init__(self, encoder, num_emotions, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden, hidden // 2)

        self.sentiment_head = nn.Linear(hidden // 2, 3)
        self.emotion_head = nn.Linear(hidden // 2, num_emotions)
        self.intensity_head = nn.Linear(hidden // 2, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]     # [CLS]

        x = torch.relu(self.shared_fc(self.dropout(h)))

        return {
            "sentiment": self.sentiment_head(x),
            "emotion": self.emotion_head(x),
            "intensity": self.intensity_head(x)
        }


class AdapterModule(nn.Module):
    """
    Lightweight adapter: down-proj -> nonlinearity -> up-proj + residual
    Applied to pooled representation (CLS) or to token outputs if desired.
    """
    def __init__(self, hidden_size:int, bottleneck:int=64, dropout:float=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # init
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.constant_(self.down.bias, 0.)
        nn.init.constant_(self.up.bias, 0.)

    def forward(self, x):
        z = self.down(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.up(z)
        return x + z  # residual


class MultiTaskBERTWithAdapters(nn.Module):
    """
    Hard-shared encoder (one transformer) + per-task adapters applied to pooled output.
    This is a practical 'adapter' implementation without external libs.
    """
    def __init__(self, transformer_name:str, num_emotions:int, adapter_bottleneck:int=64, dropout:float=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hid = self.transformer.config.hidden_size

        # task-specific adapters
        self.adapter_sentiment = AdapterModule(hid, adapter_bottleneck, dropout)
        self.adapter_emotion = AdapterModule(hid, adapter_bottleneck, dropout)
        self.adapter_intensity = AdapterModule(hid, adapter_bottleneck, dropout)

        # shared projection after adapter
        proj_dim = hid // 2
        self.shared_fc = nn.Linear(hid, proj_dim)
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(proj_dim, 3)
        self.emotion_head = nn.Linear(proj_dim, num_emotions)
        self.intensity_head = nn.Linear(proj_dim, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS pooled

        s = self.adapter_sentiment(pooled)
        e = self.adapter_emotion(pooled)
        i = self.adapter_intensity(pooled)

        # pass through shared fc
        s_x = torch.relu(self.shared_fc(self.dropout(s)))
        e_x = torch.relu(self.shared_fc(self.dropout(e)))
        i_x = torch.relu(self.shared_fc(self.dropout(i)))

        return {
            "sentiment": self.sentiment_head(s_x),
            "emotion": self.emotion_head(e_x),
            "intensity": self.intensity_head(i_x)
        }


class MMOE_Core(nn.Module):
    """
    Implementation of Mixture-of-Experts layer.
    experts: list of feed-forward networks
    gates: per-task gate producing mixture weights
    """
    def __init__(self, input_dim:int, expert_hidden:int, num_experts:int, num_tasks:int, dropout:float=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, expert_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, input_dim),
            nn.ReLU()
        ) for _ in range(num_experts)])

        # per-task gating networks
        self.gates = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        ) for _ in range(num_tasks)])

    def forward(self, x):
        # x: (B, D)
        expert_outs = []  # list (num_experts) of (B, D)
        for e in self.experts:
            expert_outs.append(e(x))  # B x D

        # stack experts -> (B, num_experts, D)
        expert_stack = torch.stack(expert_outs, dim=1)

        # For each task, compute gated sum
        task_outputs = []
        for g in self.gates:
            weights = g(x)  # (B, num_experts)
            weights = weights.unsqueeze(-1)  # (B, num_experts, 1)
            # weighted sum over experts
            task_out = (weights * expert_stack).sum(dim=1)  # (B, D)
            task_outputs.append(task_out)
        # returns list len=num_tasks of tensors (B,D)
        return task_outputs


class MultiTaskMMOE(nn.Module):
    """
    Encoder -> MMOE -> per-task head
    tasks: sentiment, emotion, intensity
    """
    def __init__(self, transformer_name:str, num_emotions:int, num_experts:int=4, expert_hidden:int=256, dropout:float=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hid = self.transformer.config.hidden_size

        self.mmoe = MMOE_Core(input_dim=hid, expert_hidden=expert_hidden, num_experts=num_experts, num_tasks=3, dropout=dropout)

        # heads after task-specific mixture outputs
        self.sent_fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.em_fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.int_fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))

        self.sentiment_head = nn.Linear(hid//2, 3)
        self.emotion_head = nn.Linear(hid//2, num_emotions)
        self.intensity_head = nn.Linear(hid//2, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # (B, hid)

        task_feats = self.mmoe(pooled)  # list len 3 of (B, hid)
        s_feat, e_feat, i_feat = task_feats

        s_x = self.sent_fc(s_feat)
        e_x = self.em_fc(e_feat)
        i_x = self.int_fc(i_feat)

        return {
            "sentiment": self.sentiment_head(s_x),
            "emotion": self.emotion_head(e_x),
            "intensity": self.intensity_head(i_x)
        }

class SingleTaskModel(nn.Module):
    def __init__(self, transformer_name: str, task: str, num_emotions: int, dropout: float = 0.3):
        super().__init__()
        self.task = task

        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        if task == "sentiment":
            self.head = nn.Linear(hid, 3)

        elif task == "emotion":
            self.head = nn.Linear(hid, num_emotions)

        elif task == "intensity":
            self.head = nn.Linear(hid, num_emotions * 3)

        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        h = self.dropout(h)
        logits = self.head(h)

        return {
            "sentiment": logits if self.task == "sentiment" else None,
            "emotion": logits if self.task == "emotion" else None,
            "intensity": logits if self.task == "intensity" else None
        }

class SoftSharingModel(nn.Module):
    """
    Soft-sharing: separate encoders but penalize divergence between encoder parameters (L2)
    We'll expose a method get_soft_sharing_loss() which computes extra reg loss.
    """
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        # Separate encoders for each task
        self.encoder_sent = AutoModel.from_pretrained(transformer_name)
        self.encoder_em = AutoModel.from_pretrained(transformer_name)
        self.encoder_int = AutoModel.from_pretrained(transformer_name)

        hid = self.encoder_sent.config.hidden_size
        proj_dim = hid // 2

        # shared-ish heads after separate encoders
        self.shared_fc = nn.Linear(hid, proj_dim)
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(proj_dim, 3)
        self.emotion_head = nn.Linear(proj_dim, num_emotions)
        self.intensity_head = nn.Linear(proj_dim, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out_s = self.encoder_sent(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        out_e = self.encoder_em(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        out_i = self.encoder_int(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]

        s = torch.relu(self.shared_fc(self.dropout(out_s)))
        e = torch.relu(self.shared_fc(self.dropout(out_e)))
        i = torch.relu(self.shared_fc(self.dropout(out_i)))

        return self.sentiment_head(s), self.emotion_head(e), self.intensity_head(i)

    def get_soft_sharing_loss(self, l2_lambda:float=1e-4):
        """
        Compute L2 distance between matching parameter tensors of encoders.
        This is a simple soft-sharing regularizer.
        """
        loss = 0.0
        # iterate over named params of one encoder and compare shapes
        for (n1,p1), (n2,p2), (n3,p3) in zip(self.encoder_sent.named_parameters(),
                                             self.encoder_em.named_parameters(),
                                             self.encoder_int.named_parameters()):
            if p1.shape == p2.shape:
                loss = loss + ((p1 - p2).pow(2).sum() + (p1 - p3).pow(2).sum())
        return l2_lambda * loss

class SoftSharingModel(nn.Module):
    """
    Soft-sharing encoder-only model.
    """
    def __init__(self, transformer_name, num_emotions, tasks, dropout=0.3):
        super().__init__()
        self.tasks = tasks

        self.encoders = nn.ModuleDict({
            task: AutoModel.from_pretrained(transformer_name)
            for task in tasks
        })

        hid = next(iter(self.encoders.values())).config.hidden_size
        proj = hid // 2

        self.shared_fc = nn.Linear(hid, proj)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        if "sentiment" in tasks:
            self.heads["sentiment"] = nn.Linear(proj, 3)
        if "emotion" in tasks:
            self.heads["emotion"] = nn.Linear(proj, num_emotions)
        if "intensity" in tasks:
            self.heads["intensity"] = nn.Linear(proj, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        outputs = {"sentiment": None, "emotion": None, "intensity": None}

        for task, encoder in self.encoders.items():
            h = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0]

            x = torch.relu(self.shared_fc(self.dropout(h)))
            outputs[task] = self.heads[task](x)

        return outputs

    def get_soft_sharing_loss(self, l2_lambda=1e-4):
        loss = 0.0
        encs = list(self.encoders.values())
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                for p1, p2 in zip(encs[i].parameters(), encs[j].parameters()):
                    if p1.shape == p2.shape:
                        loss += (p1 - p2).pow(2).sum()
        return l2_lambda * loss


class CrossStitchModel(nn.Module):
    """
    Cross-stitch networks: learn linear combination of feature maps from different tasks.
    We'll implement a simplified cross-stitch between {sent, emotion+intensity} streams.
    """
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        self.encoder_a = AutoModel.from_pretrained(transformer_name)  # e.g. sentiment
        self.encoder_b = AutoModel.from_pretrained(transformer_name)  # e.g. emotion/intensity

        hid = self.encoder_a.config.hidden_size
        self.cross_coeff = nn.Parameter(torch.eye(2))  # 2x2 cross-stitch matrix (learnable)
        self.shared_fc = nn.Linear(hid, hid//2)
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(hid//2, 3)
        self.emotion_head = nn.Linear(hid//2, num_emotions)
        self.intensity_head = nn.Linear(hid//2, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        a = self.encoder_a(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]  # BxH
        b = self.encoder_b(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]  # BxH

        # combine features via learned 2x2 matrix per batch (applied same for all dims)
        # [a'; b'] = cross_coeff @ [a; b]
        stacked = torch.stack([a, b], dim=1)  # (B, 2, H)
        # apply mixing: for each batch, for each feature dimension mixing is same matrix -> do matmul
        # Result: (B, 2, H) -> reshape -> apply
        coeff = self.cross_coeff.unsqueeze(0)  # (1,2,2)
        mixed = torch.matmul(coeff, stacked)  # (1,2,2) x (B,2,H) -> broadcasting -> (B,2,H)
        a_m = mixed[:,0,:]  # (B,H)
        b_m = mixed[:,1,:]

        # use a_m for sentiment, b_m for emotion+intensity (shared)
        s = torch.relu(self.shared_fc(self.dropout(a_m)))
        e = torch.relu(self.shared_fc(self.dropout(b_m)))
        i = torch.relu(self.shared_fc(self.dropout(b_m)))

        return self.sentiment_head(s), self.emotion_head(e), self.intensity_head(i)


class SingleTaskSentiment(nn.Module):
    def __init__(self, transformer_name:str, dropout:float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size
        self.fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hid//2, 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = self.fc(out)
        return self.head(x)

class SingleTaskEmotion(nn.Module):
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size
        self.fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hid//2, num_emotions)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = self.fc(out)
        return self.head(x)

class SingleTaskIntensity(nn.Module):
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size
        self.fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hid//2, num_emotions*3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = self.fc(out)
        return self.head(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# -------------------------
# Training
# -------------------------
# def train_epoch(model, loader, optim, scheduler, device, loss_fns, weights, config):
#     #przy multi-task learning
#     model.train()
#     running_loss = 0.0
#     all_sent_preds, all_sent_labels = [], []
#     all_emotion_preds, all_emotion_labels = [], []
#     all_int_preds, all_int_labels = [], []
#
#     accumulation_steps = config.get('gradient_accumulation_steps', 1)
#     running_batch = 0
#
#     for batch in tqdm(loader, desc="Train", leave=False):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         sentiments = batch['sentiment'].to(device)
#         emotions = batch['emotions'].to(device)
#         intensities = batch['intensities'].to(device)
#
#         s_logits, e_logits, i_logits = model(input_ids, attention_mask)
#         #outputs = model(input_ids, attention_mask)
#
#         sentiment_loss = loss_fns['sentiment'](s_logits, sentiments)
#         emotion_loss = loss_fns['emotion'](e_logits, emotions)
#
#         B = i_logits.size(0)
#         num_emotions = intensities.size(1)
#         i_logits_resh = i_logits.view(B, num_emotions, 3)
#
#         intensity_loss = 0.0
#         for j in range(num_emotions):
#             intensity_loss += loss_fns['intensity'](i_logits_resh[:, j, :], intensities[:, j])
#         intensity_loss = intensity_loss / float(num_emotions)
#
#         loss = (weights['w_sentiment'] * sentiment_loss +
#                 weights['w_emotion'] * emotion_loss +
#                 weights['w_intensity'] * intensity_loss)
#
#         # jeśli model ma get_soft_sharing_loss method => add it
#         if hasattr(model, "get_soft_sharing_loss"):
#             loss = loss + model.get_soft_sharing_loss(l2_lambda=config.get("soft_sharing_lambda", 1e-4))
#
#         # Gradient accumulation
#         loss = loss / accumulation_steps
#         loss.backward()
#
#
#         running_batch += 1
#         if running_batch % accumulation_steps == 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optim.step()
#             scheduler.step()
#             optim.zero_grad()
#
#         running_loss += loss.item() * accumulation_steps
#
#         with torch.no_grad():
#             sent_preds = torch.argmax(s_logits, dim=1).cpu().numpy()
#             all_sent_preds.extend(sent_preds.tolist())
#             all_sent_labels.extend(sentiments.cpu().numpy().tolist())
#
#             #em_preds = (torch.sigmoid(e_logits) >= 0.5).long().cpu().numpy()
#             # if e_logits.shape[1] > 1 and emotions.ndim == 2 and emotions.sum(dim=1).max() > 1:
#             #     # Multi-label case
#             #     em_preds = (torch.sigmoid(e_logits) >= 0.5).long().cpu().numpy()
#             # else:
#             #     # Single-emotion case
#             #     em_preds = torch.argmax(e_logits, dim=1).cpu().numpy().reshape(-1, 1)
#
#             # Zawsze trzymaj wymiary takie same jak emotions
#             if emotions.shape[1] > 1:
#                 # Multi-label or one-hot encoded emotion matrix (B, num_emotions)
#                 # em_preds = (torch.sigmoid(e_logits) >= 0.5).long().cpu().numpy()
#
#                 # jeśli żadna emocja nie przekroczy progu → wybierz NAJBARDZIEJ prawdopodobną
#                 probs = torch.sigmoid(e_logits)
#                 em_preds = (probs >= 0.4).long()
#                 for i in range(len(em_preds)):
#                     if em_preds[i].sum() == 0:
#                         em_preds[i, torch.argmax(probs[i])] = 1
#
#             else:
#                     # Truly single column (B,)
#                     em_preds = torch.argmax(e_logits, dim=1).cpu().numpy().reshape(-1, 1)
#
#
#             all_emotion_preds.extend(em_preds.tolist())
#             all_emotion_labels.extend(emotions.long().cpu().numpy().tolist())
#
#             int_preds = torch.argmax(i_logits_resh, dim=2).cpu().numpy()
#             all_int_preds.extend(int_preds.tolist())
#             all_int_labels.extend(intensities.cpu().numpy().tolist())
#
#         # --- Sanity check: ensure matching lengths before computing metrics ---
#     for name, preds, labels in [
#         ("sentiment", all_sent_preds, all_sent_labels),
#         ("emotion", all_emotion_preds, all_emotion_labels),
#         ("intensity", all_int_preds, all_int_labels)]:
#         if len(preds) != len(labels):
#             print(f"[Warning] {name} preds ({len(preds)}) != labels ({len(labels)}); trimming to shortest.")
#             min_len = min(len(preds), len(labels))
#             preds[:] = preds[:min_len]
#             labels[:] = labels[:min_len]
#
#     avg_loss = running_loss / len(loader)
#
#     sent_acc = accuracy_score(all_sent_labels, all_sent_preds)
#     sent_f1 = f1_score(all_sent_labels, all_sent_preds, average='macro', zero_division=0)
#
#     em_pred_flat = np.array(all_emotion_preds).reshape(-1)
#     em_label_flat = np.array(all_emotion_labels).reshape(-1)
#     em_f1_micro = f1_score(em_label_flat, em_pred_flat, average='micro', zero_division=0)
#
#     int_preds_arr = np.array(all_int_preds)
#     int_labels_arr = np.array(all_int_labels)
#     int_acc = (int_preds_arr == int_labels_arr).mean()
#
#     return {
#         "loss": avg_loss,
#         "sent_acc": sent_acc,
#         "sent_f1": sent_f1,
#         "em_f1_micro": em_f1_micro,
#         "int_acc": float(int_acc)
#     }

def train_epoch(model, loader, optim, scheduler, device, loss_fns, weights, config):
    model.train()
    running_loss = 0.0

    all_sent_preds, all_sent_labels = [], []
    all_emotion_preds, all_emotion_labels = [], []
    all_int_preds, all_int_labels = [], []

    accumulation_steps = config.get("gradient_accumulation_steps", 1)
    running_batch = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        sentiments = batch.get("sentiment")
        emotions = batch.get("emotions")
        intensities = batch.get("intensities")

        if sentiments is not None:
            sentiments = sentiments.to(device)
        if emotions is not None:
            emotions = emotions.to(device)
        if intensities is not None:
            intensities = intensities.to(device)

        outputs = model(input_ids, attention_mask)
        loss = 0.0

        # ---------- SENTIMENT ----------
        if outputs["sentiment"] is not None:
            sent_loss = loss_fns["sentiment"](outputs["sentiment"], sentiments)
            loss += weights["w_sentiment"] * sent_loss

        # ---------- EMOTION ----------
        if outputs["emotion"] is not None:
            emo_loss = loss_fns["emotion"](outputs["emotion"], emotions)
            loss += weights["w_emotion"] * emo_loss

        # ---------- INTENSITY ----------
        if outputs["intensity"] is not None:
            B = outputs["intensity"].size(0)
            num_emotions = intensities.size(1)

            i_logits = outputs["intensity"].view(B, num_emotions, 3)
            int_loss = 0.0
            for j in range(num_emotions):
                int_loss += loss_fns["intensity"](i_logits[:, j, :], intensities[:, j])
            int_loss = int_loss / float(num_emotions)

            loss += weights["w_intensity"] * int_loss

        # ---------- SOFT SHARING REG ----------
        if hasattr(model, "get_soft_sharing_loss"):
            loss += model.get_soft_sharing_loss(
                l2_lambda=config.get("soft_sharing_lambda", 1e-4)
            )

        # ---------- BACKPROP ----------
        loss = loss / accumulation_steps
        loss.backward()

        running_batch += 1
        if running_batch % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

        running_loss += loss.item() * accumulation_steps

        # ---------- METRICS (only if task exists) ----------
        with torch.no_grad():
            if outputs["sentiment"] is not None:
                preds = torch.argmax(outputs["sentiment"], dim=1).cpu().numpy()
                all_sent_preds.extend(preds.tolist())
                all_sent_labels.extend(sentiments.cpu().numpy().tolist())

            if outputs["emotion"] is not None:
                probs = torch.sigmoid(outputs["emotion"])
                em_preds = (probs >= 0.4).long()

                for i in range(len(em_preds)):
                    if em_preds[i].sum() == 0:
                        em_preds[i, torch.argmax(probs[i])] = 1

                all_emotion_preds.extend(em_preds.cpu().numpy().tolist())
                all_emotion_labels.extend(emotions.cpu().numpy().tolist())

            if outputs["intensity"] is not None:
                preds = torch.argmax(i_logits, dim=2).cpu().numpy()
                all_int_preds.extend(preds.tolist())
                all_int_labels.extend(intensities.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader)

    metrics = {"loss": avg_loss}

    if all_sent_preds:
        metrics["sent_acc"] = accuracy_score(all_sent_labels, all_sent_preds)
        metrics["sent_f1"] = f1_score(
            all_sent_labels, all_sent_preds, average="macro", zero_division=0
        )

    if all_emotion_preds:
        em_pred_flat = np.array(all_emotion_preds).reshape(-1)
        em_label_flat = np.array(all_emotion_labels).reshape(-1)
        metrics["em_f1_micro"] = f1_score(
            em_label_flat, em_pred_flat, average="micro", zero_division=0
        )

    if all_int_preds:
        int_preds_arr = np.array(all_int_preds)
        int_labels_arr = np.array(all_int_labels)
        metrics["int_acc"] = float((int_preds_arr == int_labels_arr).mean())

    return metrics

# def eval_epoch(model, loader, device, loss_fns, weights):
#     #przy multi-task learning
#     model.eval()
#     running_loss = 0.0
#     all_sent_preds, all_sent_labels = [], []
#     all_emotion_preds, all_emotion_labels = [], []
#     all_int_preds, all_int_labels = [], []
#
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Eval", leave=False):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             sentiments = batch['sentiment'].to(device)
#             emotions = batch['emotions'].to(device)
#             intensities = batch['intensities'].to(device)
#
#             s_logits, e_logits, i_logits = model(input_ids, attention_mask)
#
#             sentiment_loss = loss_fns['sentiment'](s_logits, sentiments)
#             emotion_loss = loss_fns['emotion'](e_logits, emotions)
#
#             B = i_logits.size(0)
#             num_emotions = intensities.size(1)
#             i_logits_resh = i_logits.view(B, num_emotions, 3)
#
#             intensity_loss = 0.0
#             for j in range(num_emotions):
#                 intensity_loss += loss_fns['intensity'](i_logits_resh[:, j, :], intensities[:, j])
#             intensity_loss = intensity_loss / float(num_emotions)
#
#             loss = (weights['w_sentiment'] * sentiment_loss +
#                     weights['w_emotion'] * emotion_loss +
#                     weights['w_intensity'] * intensity_loss)
#
#             running_loss += loss.item()
#
#             sent_preds = torch.argmax(s_logits, dim=1).cpu().numpy()
#             all_sent_preds.extend(sent_preds.tolist())
#             all_sent_labels.extend(sentiments.cpu().numpy().tolist())
#
#             em_preds = (torch.sigmoid(e_logits) >= 0.5).long().cpu().numpy()
#             all_emotion_preds.extend(em_preds.tolist())
#             all_emotion_labels.extend(emotions.long().cpu().numpy().tolist())
#
#             int_preds = torch.argmax(i_logits_resh, dim=2).cpu().numpy()
#             all_int_preds.extend(int_preds.tolist())
#             all_int_labels.extend(intensities.cpu().numpy().tolist())
#
#     avg_loss = running_loss / len(loader)
#
#     sent_acc = accuracy_score(all_sent_labels, all_sent_preds)
#     sent_f1 = f1_score(all_sent_labels, all_sent_preds, average='macro', zero_division=0)
#
#     em_pred_flat = np.array(all_emotion_preds).reshape(-1)
#     em_label_flat = np.array(all_emotion_labels).reshape(-1)
#     em_f1_micro = f1_score(em_label_flat, em_pred_flat, average='micro', zero_division=0)
#
#     int_preds_arr = np.array(all_int_preds)
#     int_labels_arr = np.array(all_int_labels)
#     int_acc = (int_preds_arr == int_labels_arr).mean()
#
#     metrics = {
#         "loss": avg_loss,
#         "sent_acc": sent_acc,
#         "sent_f1": sent_f1,
#         "em_f1_micro": em_f1_micro,
#         "int_acc": float(int_acc)
#     }
#
#     raw_outputs = {
#         "sent_preds": all_sent_preds,
#         "sent_labels": all_sent_labels,
#         "em_preds": all_emotion_preds,
#         "em_labels": all_emotion_labels,
#         "int_preds": all_int_preds,
#         "int_labels": all_int_labels
#     }
#
#     return metrics, raw_outputs

def eval_epoch(model, loader, device, loss_fns, weights):
    model.eval()
    running_loss = 0.0

    all_sent_preds, all_sent_labels = [], []
    all_emotion_preds, all_emotion_labels = [], []
    all_int_preds, all_int_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            sentiments = batch.get("sentiment")
            emotions = batch.get("emotions")
            intensities = batch.get("intensities")

            if sentiments is not None:
                sentiments = sentiments.to(device)
            if emotions is not None:
                emotions = emotions.to(device)
            if intensities is not None:
                intensities = intensities.to(device)

            outputs = model(input_ids, attention_mask)
            loss = 0.0

            if outputs["sentiment"] is not None:
                loss += weights["w_sentiment"] * loss_fns["sentiment"](
                    outputs["sentiment"], sentiments
                )

                preds = torch.argmax(outputs["sentiment"], dim=1).cpu().numpy()
                all_sent_preds.extend(preds.tolist())
                all_sent_labels.extend(sentiments.cpu().numpy().tolist())

            if outputs["emotion"] is not None:
                loss += weights["w_emotion"] * loss_fns["emotion"](
                    outputs["emotion"], emotions
                )

                em_preds = (torch.sigmoid(outputs["emotion"]) >= 0.5).long()
                all_emotion_preds.extend(em_preds.cpu().numpy().tolist())
                all_emotion_labels.extend(emotions.cpu().numpy().tolist())

            if outputs["intensity"] is not None:
                B = outputs["intensity"].size(0)
                num_emotions = intensities.size(1)
                i_logits = outputs["intensity"].view(B, num_emotions, 3)

                int_loss = 0.0
                for j in range(num_emotions):
                    int_loss += loss_fns["intensity"](i_logits[:, j, :], intensities[:, j])
                loss += weights["w_intensity"] * (int_loss / num_emotions)

                preds = torch.argmax(i_logits, dim=2).cpu().numpy()
                all_int_preds.extend(preds.tolist())
                all_int_labels.extend(intensities.cpu().numpy().tolist())

            running_loss += loss.item()

    metrics = {"loss": running_loss / len(loader)}

    if all_sent_preds:
        metrics["sent_acc"] = accuracy_score(all_sent_labels, all_sent_preds)
        metrics["sent_f1"] = f1_score(
            all_sent_labels, all_sent_preds, average="macro", zero_division=0
        )

    if all_emotion_preds:
        em_pred_flat = np.array(all_emotion_preds).reshape(-1)
        em_label_flat = np.array(all_emotion_labels).reshape(-1)
        metrics["em_f1_micro"] = f1_score(
            em_label_flat, em_pred_flat, average="micro", zero_division=0
        )

    if all_int_preds:
        int_preds_arr = np.array(all_int_preds)
        int_labels_arr = np.array(all_int_labels)
        metrics["int_acc"] = float((int_preds_arr == int_labels_arr).mean())

    raw_outputs = {
        "sent_preds": all_sent_preds,
        "sent_labels": all_sent_labels,
        "em_preds": all_emotion_preds,
        "em_labels": all_emotion_labels,
        "int_preds": all_int_preds,
        "int_labels": all_int_labels,
    }

    return metrics, raw_outputs

def run_pipeline(csv_path, config):
    set_seed(config['seed'])
    os.makedirs(config['output_dir'], exist_ok=True)

    print("\n" + "="*60)
    print("Loading and validating data...")
    print("="*60)

    texts, sentiments, emotions, intensities, emotion_names, mode_type = load_multitask_data(csv_path)
    print(f"\nTraining mode automatically set to: {mode_type.upper()}")

    num_emotions = len(emotion_names)

    # Split data
    try:
        X_train, X_val, y_train, y_val, e_train, e_val, i_train, i_val = train_test_split(
            texts, sentiments, emotions, intensities,
            test_size=0.2,
            random_state=config['seed'],
            stratify=sentiments
        )
    except:
        print("Stratified split failed, using random split")
        X_train, X_val, y_train, y_val, e_train, e_val, i_train, i_val = train_test_split(
            texts, sentiments, emotions, intensities,
            test_size=0.2,
            random_state=config['seed']
        )

    print(f"\nTrain: {len(X_train)} samples | Val: {len(X_val)} samples")

    print("\nEmotion distribution:")
    for i, ename in enumerate(emotion_names):
        presence = emotions[:, i].sum()
        print(f"  {ename}: {presence}/{len(emotions)} ({100*presence/len(emotions):.1f}%)")

    # --- LOAD TOKENIZER + ENCODER (shared for all architectures) ---
    if config["transformer_model"] == "bert":
        tokenizer = BertTokenizer.from_pretrained(config["transformer_model"])
        encoder = BertModel.from_pretrained(config["transformer_model"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["transformer_model"])
        encoder = AutoModel.from_pretrained(config["transformer_model"])

    train_ds = MultiTaskDataset(X_train, y_train, e_train, i_train, tokenizer, config['max_len'])
    val_ds = MultiTaskDataset(X_val, y_val, e_val, i_val, tokenizer, config['max_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    device = torch.device(config['device'])
    print(f"Device: {device}")
    emotion_freq = emotions.mean(axis=0)
    emotion_weights = torch.tensor(1/(emotion_freq+1e-6), dtype=torch.float).to(device)

    # --- SELECT MULTI-TASK ARCHITECTURE ---
    if config["model_type"] == "bert":
        model = MultiTaskBERT(
            encoder=encoder,
            num_emotions=num_emotions,
            dropout=config["dropout"]
        )
    elif config["model_type"] == "single_task":
        model = SingleTaskModel(
            transformer_name=config["transformer_model"],
            task=config["tasks"][0],
            num_emotions=num_emotions,
            dropout=config["dropout"]
        )
    elif config["model_type"] == "bert_lstm":
        model = MultiTaskBERTLSTM(
            bert_model=encoder,
            num_emotions=num_emotions,
            lstm_hidden=config["lstm_hidden_dim"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"]
        )
    elif config["model_type"] == "adapters":
        adapter_bottleneck = config.get("adapter_bottleneck", 64)
        model = MultiTaskBERTWithAdapters(
            transformer_name=config["transformer_model"],
            num_emotions=num_emotions,
            adapter_bottleneck=adapter_bottleneck,
            dropout=config["dropout"]
        )
    elif config["model_type"] == "mmoe":
        model = MultiTaskMMOE(
            transformer_name=config["transformer_model"],
            num_emotions=num_emotions,
            num_experts=config.get("num_experts", 4),
            expert_hidden=config.get("expert_hidden", 256),
            dropout=config["dropout"]
        )
    elif config["model_type"] == "soft_sharing":
        model = SoftSharingModel(
            transformer_name=config["transformer_model"],
            num_emotions=num_emotions,
            tasks=config["tasks"],
            dropout=config["dropout"]
        )
    elif config["model_type"] == "cross_stitch":
        model = CrossStitchModel(
            transformer_name=config["transformer_model"],
            num_emotions=num_emotions,
            dropout=config["dropout"]
        )
    elif config["model_type"] in ("single_sentiment","single_emotion","single_intensity"):
        if config["model_type"] == "single_sentiment":
            model = SingleTaskSentiment(transformer_name=config["transformer_model"])
        elif config["model_type"] == "single_emotion":
            model = SingleTaskEmotion(transformer_name=config["transformer_model"], num_emotions=num_emotions)
        else:
            model = SingleTaskIntensity(transformer_name=config["transformer_model"], num_emotions=num_emotions)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    model.to(device)
    # model = MultiTaskBERTLSTM(
    #     bert_model=bert_model,
    #     num_emotions=num_emotions,
    #     lstm_hidden=config['lstm_hidden_dim'],
    #     lstm_layers=config['lstm_layers'],
    #     dropout=config['dropout'],
    #     bidirectional=config['bidirectional']
    # ).to(device)

    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)

    weights_full = np.ones(3, dtype=np.float32)
    for i, c in enumerate(unique_classes):
        weights_full[c] = class_weights[i]

    class_weights_tensor = torch.tensor(weights_full, dtype=torch.float).to(device)
    tasks = config["tasks"]
    loss_fns = {}

    if "sentiment" in tasks:
        loss_fns["sentiment"] = nn.CrossEntropyLoss(weight=class_weights_tensor)

    if "emotion" in tasks:
        loss_fns["emotion"] = FocalLoss(
            alpha=emotion_weights,
            gamma=config["focal_gamma"]
        )

    if "intensity" in tasks:
        loss_fns["intensity"] = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),  # Standardowe wartości
        eps=1e-8
    )

    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\n Training setup:")
    print(f"   - Total steps: {num_training_steps}")
    print(f"   - Warmup steps: {num_warmup_steps}")
    print(f"   - Focal Loss: {config['use_focal_loss']}")
    print(f"   - Gradient accumulation: {config['gradient_accumulation_steps']}")

    # weights = {
    #     "w_sentiment": config['w_sentiment'],
    #     "w_emotion": config['w_emotion'],
    #     "w_intensity": config['w_intensity']
    # }

    weights = {}

    if "sentiment" in tasks:
        weights["w_sentiment"] = config["w_sentiment"]

    if "emotion" in tasks:
        weights["w_emotion"] = config["w_emotion"]

    if "intensity" in tasks:
        weights["w_intensity"] = config["w_intensity"]

    best_val = float('inf')
    best_epoch = 0
    history = {"train": [], "val": []}

    print("\n" + "="*60)
    print("Training...")
    print("="*60)

    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fns, weights, config)
        val_metrics, raw_outputs = eval_epoch(model, val_loader, device, loss_fns, weights)

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # print(f"Train | Loss: {train_metrics['loss']:.4f} | Sent F1: {train_metrics['sent_f1']:.4f} | Em F1: {train_metrics['em_f1_micro']:.4f} | Int Acc: {train_metrics['int_acc']:.4f}")
        # print(f"Val   | Loss: {val_metrics['loss']:.4f} | Sent F1: {val_metrics['sent_f1']:.4f} | Em F1: {val_metrics['em_f1_micro']:.4f} | Int Acc: {val_metrics['int_acc']:.4f}")

        def get_m(m, k):
            return f"{m[k]:.4f}" if k in m else "-"

        print(
            f"Train | Loss {train_metrics['loss']:.4f} | "
            f"SentF1 {get_m(train_metrics, 'sent_f1')} | "
            f"EmF1 {get_m(train_metrics, 'em_f1_micro')} | "
            f"IntAcc {get_m(train_metrics, 'int_acc')}"
        )

        # Save best model
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config['output_dir'], "best_model.pt"))
            with open(os.path.join(config['output_dir'], "best_epoch.json"), "w") as f:
                json.dump({"epoch": epoch, "val_metrics": val_metrics}, f, indent=2)
            print("✓ Saved best model")

        # Early stopping
        if epoch - best_epoch >= config['early_stopping_patience']:
            print(f"\nEarly stopping: no improvement for {config['early_stopping_patience']} epochs")
            break

    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Final Evaluation (best model)")
    print("="*60)

    model.load_state_dict(torch.load(os.path.join(config['output_dir'], "best_model.pt"), map_location=device))
    final_metrics, final_raw = eval_epoch(model, val_loader, device, loss_fns, weights)

    # Create reports directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config['output_dir'], f"reports_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    # 1. Sentiment report
    if "sentiment" in tasks:
        sent_labels_map = {0: "negative", 1: "neutral", 2: "positive"}

        # Dopasuj etykiety do rzeczywistych klas
        unique_labels = sorted(set(final_raw['sent_labels']) | set(final_raw['sent_preds']))
        target_names = [sent_labels_map[i] for i in unique_labels]

        sent_report = classification_report(
            final_raw['sent_labels'],
            final_raw['sent_preds'],
            labels=unique_labels,
            target_names=target_names,
            digits=4,
            zero_division=0
        )

        with open(os.path.join(report_dir, "sentiment_report.txt"), "w") as f:
            f.write("="*60 + "\n")
            f.write("SENTIMENT CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(sent_report)

        print("\n✓ Sentiment report saved")


    # 2. Emotion report
    if "emotion" in tasks:
        em_labels = np.array(final_raw['em_labels'])
        em_preds = np.array(final_raw['em_preds'])

        with open(os.path.join(report_dir, "emotion_report.txt"), "w") as f:
            f.write("="*60 + "\n")
            f.write("EMOTION MULTI-LABEL CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")

            em_flat_labels = em_labels.reshape(-1)
            em_flat_preds = em_preds.reshape(-1)

            em_f1_micro = f1_score(em_flat_labels, em_flat_preds, average='micro', zero_division=0)
            em_f1_macro = f1_score(em_flat_labels, em_flat_preds, average='macro', zero_division=0)
            em_acc = accuracy_score(em_flat_labels, em_flat_preds)

            f.write(f"Overall Micro F1: {em_f1_micro:.4f}\n")
            f.write(f"Overall Macro F1: {em_f1_macro:.4f}\n")
            f.write(f"Overall Accuracy: {em_acc:.4f}\n\n")
            f.write("="*60 + "\n")
            f.write("PER-EMOTION REPORTS\n")
            f.write("="*60 + "\n\n")

            for i, ename in enumerate(emotion_names):
                f.write(f"\n--- {ename.upper()} ---\n")
                try:
                    em_report = classification_report(
                        em_labels[:, i],
                        em_preds[:, i],
                        target_names=["absent", "present"],
                        digits=4,
                        zero_division=0
                    )
                    f.write(em_report + "\n")
                except Exception as e:
                    f.write(f"Error generating report: {e}\n")

        print("✓ Emotion report saved")

    # 3. Intensity
    if "intensity" in tasks:
        int_labels = np.array(final_raw['int_labels'])
        int_preds = np.array(final_raw['int_preds'])

        with open(os.path.join(report_dir, "intensity_report.txt"), "w") as f:
            f.write("="*60 + "\n")
            f.write("INTENSITY CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")

            overall_int_acc = (int_labels == int_preds).mean()
            f.write(f"Overall Intensity Accuracy: {overall_int_acc:.4f}\n\n")

            f.write("="*60 + "\n")
            f.write("PER-EMOTION INTENSITY REPORTS\n")
            f.write("="*60 + "\n\n")

            for i, ename in enumerate(emotion_names):
                f.write(f"\n--- {ename.upper()} ---\n")

                labels_i = int_labels[:, i]
                preds_i = int_preds[:, i]

                acc_i = (labels_i == preds_i).mean()
                f.write(f"Accuracy: {acc_i:.4f}\n\n")

                try:
                    int_report = classification_report(
                        labels_i,
                        preds_i,
                        target_names=["low (1)", "medium (2)", "high (3)"],
                        digits=4,
                        zero_division=0
                    )
                    f.write(int_report + "\n")
                except Exception as e:
                    f.write(f"Error: {e}\n")

        print("✓ Intensity report saved")

    # 4. Save predictions CSV
    # df_predictions = pd.DataFrame({
    #     "text": X_val,
    #     "sentiment_label": [sent_labels_map[l] for l in final_raw['sent_labels']],
    #     "sentiment_pred": [sent_labels_map[p] for p in final_raw['sent_preds']]
    # })
    #
    # for j, ename in enumerate(emotion_names):
    #     df_predictions[f"emotion__{ename}_label"] = [row[j] for row in final_raw['em_labels']]
    #     df_predictions[f"emotion__{ename}_pred"] = [row[j] for row in final_raw['em_preds']]
    #     df_predictions[f"intensity__{ename}_label"] = [row[j] + 1 for row in final_raw['int_labels']]
    #     df_predictions[f"intensity__{ename}_pred"] = [row[j] + 1 for row in final_raw['int_preds']]

    df_predictions = pd.DataFrame({"text": X_val})

    if "sentiment" in tasks:
        sent_labels_map = {0: "negative", 1: "neutral", 2: "positive"}
        df_predictions["sentiment_label"] = [sent_labels_map[l] for l in final_raw["sent_labels"]]
        df_predictions["sentiment_pred"] = [sent_labels_map[p] for p in final_raw["sent_preds"]]

    if "emotion" in tasks:
        for j, ename in enumerate(emotion_names):
            df_predictions[f"emotion__{ename}_label"] = [row[j] for row in final_raw["em_labels"]]
            df_predictions[f"emotion__{ename}_pred"] = [row[j] for row in final_raw["em_preds"]]

    if "intensity" in tasks:
        for j, ename in enumerate(emotion_names):
            df_predictions[f"intensity__{ename}_label"] = [row[j] + 1 for row in final_raw["int_labels"]]
            df_predictions[f"intensity__{ename}_pred"] = [row[j] + 1 for row in final_raw["int_preds"]]

    pred_path = os.path.join(report_dir, "predictions.csv")
    df_predictions.to_csv(pred_path, index=False, encoding='utf-8')
    print(f"✓ Predictions saved: {pred_path}")

    # 5. Save training history
    history_path = os.path.join(config['output_dir'], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")

    # 6. Final summary
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_metrics": final_metrics,
        "config": config,
        "emotion_names": emotion_names,
        "report_directory": report_dir
    }

    summary_path = os.path.join(config['output_dir'], "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val:.4f}")
    print(f"\nFinal Metrics:")
    print(f"  Sentiment F1:       {final_metrics['sent_f1']:.4f}")
    print(f"  Emotion F1 (micro): {final_metrics['em_f1_micro']:.4f}")
    print(f"  Intensity Acc:      {final_metrics['int_acc']:.4f}")
    print(f"\nReports saved to: {report_dir}")
    print("="*60)

    return summary

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    start_time = time.time()
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent.parent #C:\Users\juwieczo\DataspellProjects\meisd_project\pipeline
    csv_path = PROJECT_DIR / "EMOTIA" / "EMOTIA-DA" / "outputs22112025" / "multilabel_augmented_onehot_11222025.csv"

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    config = DEFAULT_CONFIG.copy()

    config.update({
        "model_type": "single_task", # architektura
        "transformer_model": "bert-base-uncased", # model jezykowy
        "tasks": ["sentiment"], #"emotion", "intensity" lub oba na raz do 2-task
        "output_dir": "./outputs_multitask",
        "epochs": 6, #5,
        "batch_size": 16, #16,
        "max_len": 128, #128,
        "learning_rate": 2e-5,
        "seed": 42,
        "w_sentiment": 1.0,
        "w_emotion": 2.0,
        "w_intensity": 0.7,
        "early_stopping_patience": 3,
        # NEW SETTINGS
        "use_focal_loss": True,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "warmup_ratio": 0.2, #0.1
        "gradient_accumulation_steps": 1  # Zwiększ do 4 jeśli mało GPU RAM
    })

    print("\n" + "="*60)
    print("MULTI-TASK EMOTION CLASSIFIER - IMPROVED")
    print("="*60)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:30s}: {v}")

    result = run_pipeline(csv_path, config)

    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    with open("runtime_log.txt", "a") as f:
        f.write(f"{csv_path} : {hours}h {minutes}m {seconds:.2f}s\n")
    print(f"\nCzas działania: {int(hours)}h {int(minutes)}m {seconds:.2f}s")



    print("\nAll done! Check outputs in:", config['output_dir'])
#%%
