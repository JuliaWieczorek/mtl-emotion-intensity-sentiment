import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from pathlib import Path

# =========================
# 1. LOAD DATA
# =========================

# ZMIEN SCIEZKE
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent.parent #C:\Users\juwieczo\DataspellProjects\meisd_project\pipeline
FILE_PATH = PROJECT_DIR / "data" / "MEISD_text.csv"

df = pd.read_csv(FILE_PATH)

# =========================
# 2. BASIC CLEANING
# =========================

emotion_cols = ["emotion", "emotion2", "emotion3"]
intensity_cols = ["intensity", "intensity2", "intensity3"]

# normalizacja tekstu
def normalize_label(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x in ["", "neutral", "nan"]:
        return None
    # opcjonalnie: odrzucanie oczywistych błędów
    if not x.isalpha():
        return None
    return x

for col in emotion_cols:
    df[col] = df[col].apply(normalize_label)

# =========================
# 3. EMOTION LIST PER UTTERANCE
# =========================

def extract_emotions(row):
    return [row[c] for c in emotion_cols if row[c] is not None]

df["emotions"] = df.apply(extract_emotions, axis=1)
df["num_emotions"] = df["emotions"].apply(len)

# =========================
# 4. TABLE 1: NUMBER OF EMOTIONS PER UTTERANCE
# =========================

table_num_emotions = (
    df["num_emotions"]
    .value_counts()
    .sort_index()
    .reset_index()
)

table_num_emotions.columns = ["Number_of_Emotions", "Count"]
table_num_emotions["Percentage"] = (
    table_num_emotions["Count"] / len(df) * 100
).round(2)

table_num_emotions.to_csv("table_num_emotions.csv", index=False)

# =========================
# 5. TABLE 2: MULTILABEL SUMMARY (PAPER-READY)
# =========================

table_multilabel = pd.DataFrame({
    "Metric": [
        "Total utterances",
        "Neutral only (0 emotions)",
        "Single emotion",
        "Multilabel (≥2 emotions)",
        "Three emotions"
    ],
    "Value": [
        len(df),
        (df["num_emotions"] == 0).sum(),
        (df["num_emotions"] == 1).sum(),
        (df["num_emotions"] >= 2).sum(),
        (df["num_emotions"] == 3).sum()
    ]
})

table_multilabel["Percentage"] = (
    table_multilabel["Value"] / len(df) * 100
).round(2)

table_multilabel.to_csv("table_multilabel_summary.csv", index=False)

# =========================
# 6. TABLE 3: MOST FREQUENT EMOTION PAIRS
# =========================

pair_counter = Counter()

for emotions in df["emotions"]:
    if len(emotions) >= 2:
        for pair in combinations(sorted(emotions), 2):
            pair_counter[pair] += 1

table_pairs = (
    pd.DataFrame(pair_counter.items(), columns=["Emotion_Pair", "Count"])
    .sort_values("Count", ascending=False)
)

table_pairs["Emotion_Pair"] = table_pairs["Emotion_Pair"].apply(
    lambda x: " + ".join(x)
)

table_pairs["Percentage"] = (
    table_pairs["Count"] / len(df) * 100
).round(2)

table_pairs.to_csv("table_emotion_pairs.csv", index=False)

# =========================
# 7. TABLE 4: SENTIMENT vs MULTILABEL
# =========================

table_sentiment = (
    df.groupby("sentiment")
    .agg(
        Avg_Num_Emotions=("num_emotions", "mean"),
        Multilabel_Percentage=("num_emotions", lambda x: (x >= 2).mean() * 100)
    )
    .reset_index()
)

table_sentiment["Avg_Num_Emotions"] = table_sentiment["Avg_Num_Emotions"].round(2)
table_sentiment["Multilabel_Percentage"] = table_sentiment["Multilabel_Percentage"].round(2)

table_sentiment.to_csv("table_sentiment_multilabel.csv", index=False)

# =========================
# 8. TABLE 5: INTENSITY vs NUMBER OF EMOTIONS
# =========================

# zamiana intensywnosci na numeric
for col in intensity_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def mean_intensity(row):
    values = [row[c] for c in intensity_cols if not pd.isna(row[c])]
    return np.mean(values) if values else np.nan

df["mean_intensity"] = df.apply(mean_intensity, axis=1)

table_intensity = (
    df.groupby("num_emotions")
    .agg(
        Avg_Intensity=("mean_intensity", "mean"),
        Max_Intensity=("mean_intensity", "max"),
        Count=("mean_intensity", "count")
    )
    .reset_index()
)

table_intensity[["Avg_Intensity", "Max_Intensity"]] = table_intensity[
    ["Avg_Intensity", "Max_Intensity"]
].round(2)

table_intensity.to_csv("table_intensity_vs_emotions.csv", index=False)

# =========================
# 9. TABLE 6: EMOTION FREQUENCY (IMBALANCE ANALYSIS)
# =========================

all_emotions = [e for sub in df["emotions"] for e in sub]

table_emotion_freq = (
    pd.Series(all_emotions)
    .value_counts()
    .reset_index()
)

table_emotion_freq.columns = ["Emotion", "Frequency"]
table_emotion_freq["Percentage"] = (
    table_emotion_freq["Frequency"] / len(df) * 100
).round(2)

table_emotion_freq.to_csv("table_emotion_frequency.csv", index=False)

print("✅ All tables generated successfully.")
