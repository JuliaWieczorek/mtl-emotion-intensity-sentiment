import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
csv_path = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/EMOTIA/EMOTIA-DA/outputs22112025/MEISD_balanced_expanded.csv"

# === LOAD DATA ===
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {len(df)} samples\n")

# === AUTO-DETECT EMOTION / INTENSITY COLUMNS ===
emotion_cols = [c for c in df.columns if c.startswith("emotion")]
intensity_cols = [c for c in df.columns if c.startswith("intensity")]

print("Emotion columns:", emotion_cols)
print("Intensity columns:", intensity_cols, "\n")

# Extract unique emotions appearing in emotion1/2/3
emotions = sorted(set(df[emotion_cols].fillna("").values.flatten()) - {""})
print(f"Detected emotion types: {emotions}\n")

# === SENTIMENT DISTRIBUTION ===
if "sentiment" in df.columns:
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts(), "\n")

# === EMOTION PRESENCE COUNTS ===
emotion_presence = {emo: 0 for emo in emotions}

for emo in emotions:
    for col in emotion_cols:
        emotion_presence[emo] += (df[col] == emo).sum()

emotion_counts = pd.Series(emotion_presence).sort_values(ascending=False)

print("Emotion presence counts:")
print(emotion_counts)

# MULTI-EMOTION STATS
df["num_emotions"] = df[emotion_cols].apply(lambda row: row.notna().sum(), axis=1)
multi_ratio = (df["num_emotions"] > 1).mean() * 100

print(f"\nAverage number of emotions per sample: {df['num_emotions'].mean():.2f}")
print(f"Samples with >1 emotion: {multi_ratio:.2f}%")

# === INTENSITY DISTRIBUTION PER EMOTION ===
print("\nEmotion × Intensity distribution:")

intensity_summary = {emo: {} for emo in emotions}

for emo in emotions:
    for e_col, i_col in zip(emotion_cols, intensity_cols):
        mask = df[e_col] == emo
        values = df.loc[mask, i_col].dropna().astype(int)
        for val in values:
            intensity_summary[emo][val] = intensity_summary[emo].get(val, 0) + 1

intensity_df = pd.DataFrame(intensity_summary).fillna(0).astype(int)
print(intensity_df)

# === TOTAL INTENSITY ACROSS ALL EMOTIONS ===
all_intensities = []

for e_col, i_col in zip(emotion_cols, intensity_cols):
    vals = df[i_col].dropna().astype(int).tolist()
    all_intensities.extend(vals)

all_intensity_counts = pd.Series(all_intensities).value_counts().sort_index()

print("\nOverall intensity counts:")
print(all_intensity_counts)

# === SAVE REPORT ===
output_path = "emotion_intensity_report.csv"
intensity_df.to_csv(output_path)
print(f"\nReport saved to: {output_path}")

# === PLOTS ===
plt.figure(figsize=(8,4))
emotion_counts.plot(kind="bar", color="steelblue", title="Emotion presence counts")
plt.ylabel("Number of samples")
plt.show()

plt.figure(figsize=(6,4))
all_intensity_counts.plot(kind="bar", color="seagreen", title="Overall intensity distribution")
plt.ylabel("Count")
plt.show()
