"""
Skrypt do analizy wyników treningu
Uruchom PO zakończeniu treningu modelu
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# KONFIGURACJA
# -------------------------
OUTPUT_DIR = "./outputs_multitask"
LATEST_REPORT = None

# -------------------------
# Znajdź najnowszy folder z raportami
# -------------------------
def find_latest_report_dir(output_dir):
    """Znajdź najnowszy folder reports_XXXXXX"""
    report_dirs = [d for d in Path(output_dir).iterdir()
                   if d.is_dir() and d.name.startswith("reports_")]

    if not report_dirs:
        raise FileNotFoundError(f"Brak folderów z raportami w {output_dir}")

    # Sortuj po dacie w nazwie
    latest = sorted(report_dirs, key=lambda x: x.name)[-1]
    return str(latest)

# -------------------------
# 1. WYKRESY HISTORII TRENINGU
# -------------------------
def plot_training_history(history_path, save_dir):
    """Wykresy loss i metryk z treningu"""
    print("\n" + "="*60)
    print("Tworzenie wykresów historii treningu...")
    print("="*60)

    with open(history_path, 'r') as f:
        history = json.load(f)

    train_data = history['train']
    val_data = history['val']

    epochs = range(1, len(train_data) + 1)

    # Utwórz subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, [e['loss'] for e in train_data], 'o-', label='Train Loss', linewidth=2)
    ax.plot(epochs, [e['loss'] for e in val_data], 's-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sentiment F1
    ax = axes[0, 1]
    ax.plot(epochs, [e['sent_f1'] for e in train_data], 'o-', label='Train Sentiment F1', linewidth=2)
    ax.plot(epochs, [e['sent_f1'] for e in val_data], 's-', label='Val Sentiment F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Sentiment F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 3. Emotion F1 (micro)
    ax = axes[1, 0]
    ax.plot(epochs, [e['em_f1_micro'] for e in train_data], 'o-', label='Train Emotion F1', linewidth=2)
    ax.plot(epochs, [e['em_f1_micro'] for e in val_data], 's-', label='Val Emotion F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score (Micro)')
    ax.set_title('Emotion F1 Score (Micro)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 4. Intensity Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, [e['int_acc'] for e in train_data], 'o-', label='Train Intensity Acc', linewidth=2)
    ax.plot(epochs, [e['int_acc'] for e in val_data], 's-', label='Val Intensity Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Intensity Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    # Zapisz
    save_path = os.path.join(save_dir, "training_history_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Zapisano: {save_path}")

    plt.close()

# -------------------------
# 2. CONFUSION MATRIX dla SENTIMENT
# -------------------------
def plot_sentiment_confusion_matrix(predictions_path, save_dir):
    """Confusion matrix dla sentiment classification"""
    print("\n" + "="*60)
    print("Tworzenie confusion matrix dla sentimentu...")
    print("="*60)

    df = pd.read_csv(predictions_path)

    # Mapowanie
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y_true = df['sentiment_label'].map(label_map)
    y_pred = df['sentiment_pred'].map(label_map)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Znormalizowana wersja
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Zwykła
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                ax=axes[0])
    axes[0].set_title('Sentiment Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Znormalizowana
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                ax=axes[1])
    axes[1].set_title('Sentiment Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()

    save_path = os.path.join(save_dir, "sentiment_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Zapisano: {save_path}")

    plt.close()

# -------------------------
# 3. ANALIZA EMOCJI - per emotion performance
# -------------------------
def plot_emotion_performance(predictions_path, save_dir):
    """Wykres F1 score dla każdej emocji osobno"""
    print("\n" + "="*60)
    print("Analiza wydajności per-emotion...")
    print("="*60)

    df = pd.read_csv(predictions_path)

    # Znajdź kolumny z emocjami
    emotion_cols = [c for c in df.columns if c.startswith('emotion__') and c.endswith('_label')]
    emotion_names = [c.replace('emotion__', '').replace('_label', '') for c in emotion_cols]

    results = []

    for ename in emotion_names:
        label_col = f'emotion__{ename}_label'
        pred_col = f'emotion__{ename}_pred'

        y_true = df[label_col].values
        y_pred = df[pred_col].values

        # Accuracy
        acc = (y_true == y_pred).mean()

        # F1 score
        from sklearn.metrics import f1_score, precision_score, recall_score

        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)

        # Wskaźnik obecności
        presence = y_true.sum() / len(y_true)

        results.append({
            'emotion': ename,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': acc,
            'presence_rate': presence
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. F1, Precision, Recall
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.25

    ax.bar(x - width, results_df['f1'], width, label='F1 Score', alpha=0.8)
    ax.bar(x, results_df['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width, results_df['recall'], width, label='Recall', alpha=0.8)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Score')
    ax.set_title('Per-Emotion Performance Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['emotion'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # 2. Presence rate vs F1
    ax = axes[1]
    scatter = ax.scatter(results_df['presence_rate'], results_df['f1'],
                         s=200, alpha=0.6, c=results_df['f1'], cmap='viridis')

    for idx, row in results_df.iterrows():
        ax.annotate(row['emotion'],
                    (row['presence_rate'], row['f1']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)

    ax.set_xlabel('Presence Rate in Dataset')
    ax.set_ylabel('F1 Score')
    ax.set_title('Emotion F1 vs Presence Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='F1 Score')

    plt.tight_layout()

    save_path = os.path.join(save_dir, "emotion_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Zapisano: {save_path}")

    # Zapisz tabelę
    table_path = os.path.join(save_dir, "emotion_performance_table.csv")
    results_df.to_csv(table_path, index=False)
    print(f"✓ Zapisano tabelę: {table_path}")

    plt.close()

    return results_df

# -------------------------
# 4. ANALIZA INTENSITY
# -------------------------
def plot_intensity_confusion_matrices(predictions_path, save_dir):
    """Confusion matrices dla intensity każdej emocji"""
    print("\n" + "="*60)
    print("Tworzenie confusion matrices dla intensity...")
    print("="*60)

    df = pd.read_csv(predictions_path)

    # Znajdź emocje
    intensity_cols = [c for c in df.columns if c.startswith('intensity__') and c.endswith('_label')]
    emotion_names = [c.replace('intensity__', '').replace('_label', '') for c in intensity_cols]

    # Oblicz ile będziemy mieli subplotów
    n_emotions = len(emotion_names)
    n_cols = 3
    n_rows = (n_emotions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_emotions > 1 else [axes]

    for idx, ename in enumerate(emotion_names):
        label_col = f'intensity__{ename}_label'
        pred_col = f'intensity__{ename}_pred'

        y_true = df[label_col].values
        y_pred = df[pred_col].values

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        ax = axes[idx]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Low', 'Med', 'High'],
                    yticklabels=['Low', 'Med', 'High'],
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f'{ename}', fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # Ukryj puste subploty
    for idx in range(n_emotions, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Intensity Confusion Matrices (Normalized)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, "intensity_confusion_matrices.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Zapisano: {save_path}")

    plt.close()

# -------------------------
# 5. PODSUMOWANIE TEKSTOWE
# -------------------------
def create_summary_report(report_dir, save_dir):
    """Stwórz czytelne podsumowanie wszystkich metryk"""
    print("\n" + "="*60)
    print("Tworzenie podsumowania tekstowego...")
    print("="*60)

    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("FINAL RESULTS SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append("")

    # 1. Sentiment
    sent_file = os.path.join(report_dir, "sentiment_report.txt")
    if os.path.exists(sent_file):
        with open(sent_file, 'r') as f:
            content = f.read()
            summary_lines.append("--- SENTIMENT CLASSIFICATION ---")
            summary_lines.append(content)
            summary_lines.append("")

    # 2. Emotion
    em_file = os.path.join(report_dir, "emotion_report.txt")
    if os.path.exists(em_file):
        with open(em_file, 'r') as f:
            content = f.read()
            # Weź tylko overall metrics
            lines = content.split('\n')
            summary_lines.append("--- EMOTION CLASSIFICATION (OVERALL) ---")
            for line in lines[:10]:  # Pierwsze 10 linii
                summary_lines.append(line)
            summary_lines.append("")

    # 3. Intensity
    int_file = os.path.join(report_dir, "intensity_report.txt")
    if os.path.exists(int_file):
        with open(int_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            summary_lines.append("--- INTENSITY CLASSIFICATION (OVERALL) ---")
            for line in lines[:6]:
                summary_lines.append(line)
            summary_lines.append("")

    summary_lines.append("="*60)
    summary_lines.append("END OF SUMMARY")
    summary_lines.append("="*60)

    # Zapisz
    summary_path = os.path.join(save_dir, "SUMMARY_READABLE.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"✓ Zapisano: {summary_path}")

    # Wyświetl w konsoli
    print("\n" + "\n".join(summary_lines))

# -------------------------
# MAIN
# -------------------------
def main():
    print("\n" + "="*60)
    print("ANALIZA WYNIKÓW TRENINGU")
    print("="*60)

    # Znajdź folder z raportami
    if LATEST_REPORT is None:
        report_dir = find_latest_report_dir(OUTPUT_DIR)
    else:
        report_dir = LATEST_REPORT

    print(f"\nAnalizuję folder: {report_dir}")

    # Ścieżki do plików
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    predictions_path = os.path.join(report_dir, "predictions.csv")

    # Sprawdź czy pliki istnieją
    if not os.path.exists(history_path):
        print(f"Brak pliku: {history_path}")
        return

    if not os.path.exists(predictions_path):
        print(f"Brak pliku: {predictions_path}")
        return

    # Folder na wykresy
    plots_dir = os.path.join(report_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"📊 Wykresy będą zapisane w: {plots_dir}")

    # Wykonaj analizy
    try:
        plot_training_history(history_path, plots_dir)
        plot_sentiment_confusion_matrix(predictions_path, plots_dir)
        emotion_perf = plot_emotion_performance(predictions_path, plots_dir)
        plot_intensity_confusion_matrices(predictions_path, plots_dir)
        create_summary_report(report_dir, plots_dir)

        print("\n" + "="*60)
        print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
        print("="*60)
        print(f"\nWszystkie wyniki w: {plots_dir}")
        print("\nWygenerowane pliki:")
        print("  1. training_history_plot.png - wykresy loss/metryki")
        print("  2. sentiment_confusion_matrix.png - macierz pomyłek sentiment")
        print("  3. emotion_performance.png - wydajność każdej emocji")
        print("  4. emotion_performance_table.csv - tabela z metrykami")
        print("  5. intensity_confusion_matrices.png - macierze dla intensity")
        print("  6. SUMMARY_READABLE.txt - podsumowanie tekstowe")

    except Exception as e:
        print(f"\nBłąd podczas analizy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()