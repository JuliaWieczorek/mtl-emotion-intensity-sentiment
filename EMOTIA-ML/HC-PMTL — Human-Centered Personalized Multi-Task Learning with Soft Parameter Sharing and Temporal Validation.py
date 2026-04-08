"""Since explicit speaker identifiers are not available, we construct pseudo-user personas using
(a) linguistic style clustering and
(b) dialogue structure grouping. We further introduce a hybrid persona representation combining both signals,
enabling human-centered personalization and memory modeling."""


"""
- multitask
- soft-sharing BERT
- personalization
- user-history memory
- calibration layer
- explainable heads
- temporal validation
"""

""" bazed on  Kazienko et. al. (2023)
human-centered	        text + user modeling
personalized	        user embeddings
personal calibration	scale+bias layer
multi-task              sent + emo + int
soft-sharing            3 encodery + L2
history memory	        rolling embedding memory
reasoning-compatible	separate encoders
explainable	            token attentions
temporal validation	    past/present/future
ablation-ready	        config switches
paper metrics	        macro/micro/user
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, deque

# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    "model": "bert-base-uncased",
    "max_len": 128,
    "batch": 16,
    "epochs": 5,
    "lr": 2e-5,

    "use_soft_sharing": True,
    "use_personalization": True,
    "use_history": True,
    "use_calibration": True,

    "history_size": 5,
    "n_intensity": 4  # 0–3
}

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
emo2id = {e: i for i, e in enumerate(EMOTIONS)}
N_EMO = len(EMOTIONS)

# =========================================================
# TIME PARSING (MEISD)
# =========================================================

def parse_time_to_ms(t):
    """
    Robust parser for MEISD time formats:
    - MM:SS:ms
    - HH:MM:SS:ms
    """
    if pd.isna(t):
        return np.nan

    parts = str(t).strip().split(":")

    try:
        parts = [int(p) for p in parts]

        if len(parts) == 3:
            m, s, ms = parts
            total_ms = (m * 60 + s) * 1000 + ms

        elif len(parts) == 4:
            h, m, s, ms = parts
            total_ms = ((h * 60 + m) * 60 + s) * 1000 + ms

        else:
            return np.nan

        return total_ms

    except ValueError:
        return np.nan

def temporal_split(df, col):
    df = df.sort_values(col)
    n = len(df)
    return df[:int(.7*n)], df[int(.7*n):int(.85*n)], df[int(.85*n):]

# =========================================================
# PSEUDO USER BUILDER
# =========================================================

class PseudoUserBuilder:
    def __init__(self, n_clusters=40):
        self.n_clusters = n_clusters

    def style_persona(self, texts):
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        X = enc.encode(texts.tolist(), normalize_embeddings=True)
        return KMeans(self.n_clusters, n_init=10, random_state=42).fit_predict(X)

    def dialog_persona(self, df):
        key = df[["TV Series", "seasons", "episodes", "dialog_ids"]].astype(str).agg("_".join, axis=1)
        return LabelEncoder().fit_transform(key)

    def add(self, df):
        s = self.style_persona(df["Utterances"])
        d = self.dialog_persona(df)
        combo = pd.Series(s.astype(str)) + "_" + pd.Series(d.astype(str))
        df["pseudo_user"] = LabelEncoder().fit_transform(combo)
        return df

# =========================================================
# LABEL BUILDERS
# =========================================================

def build_emotion_vector(row):
    vec = np.zeros(N_EMO, dtype=np.float32)
    for col in ["emotion", "emotion2", "emotion3"]:
        e = row.get(col)
        if pd.notna(e) and e in emo2id:
            vec[emo2id[e]] = 1
    return vec

def build_intensity_matrix(row):
    """
    Intensity as emotion-dependent classification.
    Invalid / non-numeric labels are ignored (set to -1).
    """
    vec = np.full(N_EMO, -1, dtype=np.int64)

    for e_col, i_col in zip(
        ["emotion", "emotion2", "emotion3"],
        ["intensity", "intensity2", "intensity3"]
    ):
        e = row.get(e_col)
        i = row.get(i_col)

        if pd.isna(e) or pd.isna(i):
            continue

        if e not in emo2id:
            continue

        try:
            i_val = int(i)
        except (ValueError, TypeError):
            # handles 'neu', 'df', etc.
            continue

        vec[emo2id[e]] = i_val

    return vec

# =========================================================
# DATASET
# =========================================================

class HCDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tok = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        enc = self.tok(
            r["Utterances"],
            truncation=True,
            padding="max_length",
            max_length=CONFIG["max_len"],
            return_tensors="pt"
        )

        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "sent": torch.tensor(r["sentiment"], dtype=torch.long),
            "emo": torch.tensor(r["emo_vec"], dtype=torch.float),
            "int": torch.tensor(r["int_vec"], dtype=torch.long),
            "user": torch.tensor(r["user_id"], dtype=torch.long)
        }

# =========================================================
# USER HISTORY MEMORY
# =========================================================

class UserHistoryMemory:
    def __init__(self, size):
        self.mem = defaultdict(lambda: deque(maxlen=size))

    def push(self, u, v):
        self.mem[u].append(v.detach().cpu())

    def get(self, u, dim, device):
        if len(self.mem[u]) == 0:
            return torch.zeros(dim, device=device)
        return torch.stack(list(self.mem[u])).mean(0).to(device)

# =========================================================
# CALIBRATION
# =========================================================

class PersonalCalibration(nn.Module):
    def __init__(self, n_users, dim):
        super().__init__()
        self.scale = nn.Embedding(n_users, dim)
        self.bias = nn.Embedding(n_users, dim)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x, u):
        return x * self.scale(u) + self.bias(u)

# =========================================================
# EXPLANATION HEAD
# =========================================================

class ExplanationHead(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.att = nn.Linear(hid, 1)

    def forward(self, h):
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        return pooled, w

# =========================================================
# MODEL
# =========================================================

class HCModel(nn.Module):
    def __init__(self, n_users):
        super().__init__()

        self.enc_s = AutoModel.from_pretrained(CONFIG["model"])
        self.enc_e = AutoModel.from_pretrained(CONFIG["model"])
        self.enc_i = AutoModel.from_pretrained(CONFIG["model"])

        hid = self.enc_s.config.hidden_size
        self.expl = ExplanationHead(hid)

        self.user_emb = nn.Embedding(n_users, 32)
        self.cal = PersonalCalibration(n_users, hid)

        self.fc = nn.Linear(hid + 32, hid // 2)

        self.h_sent = nn.Linear(hid // 2, 3)
        self.h_emo = nn.Linear(hid // 2, N_EMO)
        self.h_int = nn.Linear(hid // 2, N_EMO * CONFIG["n_intensity"])

    def encode(self, enc, ids, mask, user, mem):
        h = enc(ids, mask).last_hidden_state
        pooled, _ = self.expl(h)

        hist = torch.stack([
            mem.get(u.item(), pooled.size(1), pooled.device)
            for u in user
        ])

        pooled = pooled + hist
        pooled = self.cal(pooled, user)

        return pooled

    def forward(self, ids, mask, user, mem):
        s = self.encode(self.enc_s, ids, mask, user, mem)
        e = self.encode(self.enc_e, ids, mask, user, mem)
        i = self.encode(self.enc_i, ids, mask, user, mem)

        def fuse(x):
            return torch.relu(self.fc(torch.cat([x, self.user_emb(user)], 1)))

        s, e, i = map(fuse, (s, e, i))

        return {
            "sent": self.h_sent(s),
            "emo": self.h_emo(e),
            "int": self.h_int(i).view(-1, N_EMO, CONFIG["n_intensity"]),
            "repr": s.detach()
        }

    def soft_loss(self):
        if not CONFIG["use_soft_sharing"]:
            return 0
        loss = 0
        encs = [self.enc_s, self.enc_e, self.enc_i]
        for a in range(3):
            for b in range(a+1, 3):
                for p1, p2 in zip(encs[a].parameters(), encs[b].parameters()):
                    if p1.shape == p2.shape:
                        loss += (p1 - p2).pow(2).sum()
        return 1e-4 * loss

# =========================================================
# TRAIN
# =========================================================

def train_epoch(model, loader, opt, mem, device):
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    model.train()

    for b in loader:
        ids = b["ids"].to(device)
        mask = b["mask"].to(device)
        user = b["user"].to(device)

        out = model(ids, mask, user, mem)

        loss = ce(out["sent"], b["sent"].to(device))
        loss += bce(out["emo"], b["emo"].to(device))

        # intensity: ONLY for active emotions
        for j in range(N_EMO):
            mask_j = b["int"][:, j] >= 0
            if mask_j.any():
                loss += ce(
                    out["int"][mask_j, j],
                    b["int"][mask_j, j].to(device)
                )

        loss += model.soft_loss()

        loss.backward()
        opt.step()
        opt.zero_grad()

        for u, v in zip(user.tolist(), out["repr"]):
            mem.push(u, v)

# =========================================================
# RUN
# =========================================================
SENT_MAP = {
        "negative": 0, "neg": 0, "0": 0, 0: 0,
        "neutral": 1, "neu": 1, "1": 1, 1: 1,
        "positive": 2, "pos": 2, "2": 2, 2: 2
    }
def run(csv):
    df = pd.read_csv(csv)

    df["timestamp"] = df["start_times"].apply(parse_time_to_ms)

    builder = PseudoUserBuilder()
    df = builder.add(df)

    df["emo_vec"] = df.apply(build_emotion_vector, axis=1)
    df["int_vec"] = df.apply(build_intensity_matrix, axis=1)
    df["sentiment"] = df["sentiment"].map(SENT_MAP)
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    users = df["pseudo_user"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    df["user_id"] = df["pseudo_user"].map(user_map)

    train_df, _, _ = temporal_split(df, "timestamp")

    tok = AutoTokenizer.from_pretrained(CONFIG["model"])
    ds = HCDataset(train_df, tok)
    loader = DataLoader(ds, batch_size=CONFIG["batch"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HCModel(len(users)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    mem = UserHistoryMemory(CONFIG["history_size"])

    for ep in range(CONFIG["epochs"]):
        train_epoch(model, loader, opt, mem, device)
        print(f"epoch {ep} done")

csv = "C:/Users/Julixus/DataspellProjects/meisd_project/data/MEISD_text.csv"
run(csv)