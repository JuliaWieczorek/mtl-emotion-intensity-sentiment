# ============================================================
# analyze_balance_requirements.py
# ------------------------------------------------------------
# Utility to analyze ESConv dataset distribution (emotion × intensity)
# and estimate augmentation parameters:
# - how many samples are needed to balance (num_samples)
# - how much to expand the dataset (expand_percent)
# ============================================================

import pandas as pd
from pathlib import Path

# === CONFIG ===
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent.parent
ESCONV_PATH = PROJECT_DIR / "data" / "MEISD_DA_ready.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "MEISD_balance_analysis_report.csv"

# === LOAD DATA ===
print(f"Loading ESConv dataset from: {ESCONV_PATH}")
df = pd.read_csv(ESCONV_PATH)
print(f"Loaded {len(df)} samples\n")

# === CLEAN AND PREPARE ===
df['emotion1'] = df['emotion1'].astype(str).str.lower().str.strip()
df['intensity1'] = df['intensity1'].astype(float)

# === GROUP BY (emotion1, intensity1) ===
group_counts = df.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
group_counts = group_counts.sort_values('count', ascending=False).reset_index(drop=True)

max_count = group_counts['count'].max()
needed_total = (max_count - group_counts['count']).sum()

print("=== Current Distribution (Top 15) ===")
print(group_counts.head(15))
print(f"\nMaximum samples per (emotion, intensity): {max_count}")
print(f"Total additional samples needed to balance all groups: {needed_total}\n")

# === ESTIMATE PARAMETERS ===
num_samples_recommend = int(needed_total)
expand_percent_recommend = 70  # default suggestion

print("=== Suggested Parameters ===")
print(f"Recommended num_samples  : {num_samples_recommend}")
print(f"Recommended expand_percent: {expand_percent_recommend}%")

# === OPTIONAL: SAVE REPORT ===
group_counts['needed_to_balance'] = max_count - group_counts['count']
group_counts.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print(f"\nDetailed balance report saved to: {OUTPUT_PATH}")
print("You can open it in Excel or pandas to inspect per-class details.\n")

# === OPTIONAL SUMMARY ===
summary = {
    "total_original_samples": len(df),
    "unique_combinations": len(group_counts),
    "max_per_group": max_count,
    "needed_to_balance": needed_total,
    "recommended_num_samples": num_samples_recommend,
    "recommended_expand_percent": expand_percent_recommend
}

summary_path = PROJECT_DIR / "data" / "balance_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print(f"Summary saved to: {summary_path}")
print("\nAnalysis complete! Use these parameters in your augmentation pipeline.\n")





# Analiza strategiczna: MEISD vs ESConvŚwietne pytanie! Skoro targetem jest ESConv, to musisz dopasować MEISD do jego struktury emocjonalnej.📊 Porównanie strukturESConv (TARGET):
#
# Emocje: anxiety, depression, sadness, anger, fear, shame, disgust, nervousness, pain, jealousy, guilt
# Intensity: głównie 1.0-2.0 (brak 3.0!)
# Dominujące: anxiety, depression, sadness
# MEISD (SOURCE):
#
# Emocje: joy, anger, sadness, neutral, acceptance, surprise, disgust, fear
# Intensity: 1.0-3.0 (ma 3.0!)
# Dominujące: joy, anger, sadness, neutral
# Problem: Niezgodność emocjonalna
#
# EMOCJE W MEISD, KTÓRYCH NIE MA W ESCONV:
# ✗ joy          (514 próbek) ← nie ma w ESConv!
# ✗ neutral      (438 próbek) ← nie ma w ESConv!
# ✗ acceptance   (183 próbek) ← nie ma w ESConv!
# ✗ surprise     (139 próbek) ← nie ma w ESConv!
#
# WSPÓLNE EMOCJE (możesz wykorzystać):
# ✓ anger        (406 próbek) ← masz w ESConv
# ✓ sadness      (337 próbek) ← masz w ESConv
# ✓ disgust      (124 próbek) ← masz w ESConv
# ✓ fear         (69 próbek)  ← masz w ESConv
# ```
#
# ## Dodatkowy problem: Intensity 3.0
# ```
# INTENSITY 3.0 W MEISD (nie ma w ESConv!):
# neutral,3.0,57     ← usuń
# joy,3.0,29         ← usuń
# anger,3.0,49       ← usuń (ale anger jest w ESConv)
# surprise,3.0,17    ← usuń
# sadness,3.0,15     ← usuń (ale sadness jest w ESConv)
# fear,3.0,7         ← usuń (ale fear jest w ESConv)
# disgust,3.0,6      ← usuń (ale disgust jest w ESConv)
# acceptance,3.0,5   ← usuń