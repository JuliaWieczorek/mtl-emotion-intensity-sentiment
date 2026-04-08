import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, levene, shapiro
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(data_dict=None, csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(data_dict)

    print("=" * 80)
    print("KROK 1: PRZEGLĄD DANYCH")
    print("=" * 80)
    print(f"\nWymiary danych: {df.shape}")
    print(f"\nKolumny: {df.columns.tolist()}")
    print(f"\nPierwsze wiersze:\n{df.head()}")
    print(f"\nBrakujące wartości:\n{df.isnull().sum()}")
    print(f"\nStatystyki opisowe:\n{df.describe()}")

    return df

def aggregate_metrics(df, group_cols=['architecture', 'encoder']):
    print("\n" + "=" * 80)
    print("KROK 2: AGREGACJA METRYK")
    print("=" * 80)

    agg_df = df.groupby(group_cols).apply(
        lambda x: pd.Series({
            'weighted_f1': np.average(x['f1 score'], weights=x['support']),
            'mean_precision': x['precision'].mean(),
            'mean_recall': x['recall'].mean(),
            'mean_f1': x['f1 score'].mean(),
            'std_f1': x['f1 score'].std(),
            'total_support': x['support'].sum()
        })
    ).reset_index()

    agg_df = agg_df.sort_values('weighted_f1', ascending=False)

    print("\nZagregowane metryki (posortowane według weighted F1):")
    print(agg_df.to_string(index=False))

    return agg_df


def correlation_analysis(df, metrics=['precision', 'recall', 'f1 score']):
    print("\n" + "=" * 80)
    print("KROK 3: ANALIZA KORELACJI")
    print("=" * 80)

    # Macierz korelacji
    corr_matrix = df[metrics].corr()
    print("\nMacierz korelacji Pearsona:")
    print(corr_matrix)

    # Wizualizacja
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Macierz Korelacji Metryk')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n Zapisano wykres: correlation_matrix.png")

    return corr_matrix


def test_assumptions(df, metric='f1 score', group_col='architecture'):
    print("\n" + "=" * 80)
    print("KROK 4: TESTY ZAŁOŻEŃ")
    print("=" * 80)

    groups = df[group_col].unique()

    print(f"\n4a. Test Normalności (Shapiro-Wilk) dla '{metric}':")
    print("-" * 60)
    normality_results = {}

    for group in groups:
        data = df[df[group_col] == group][metric]
        stat, p_value = shapiro(data)
        normality_results[group] = {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        print(f"{group:15s}: statystyka={stat:.4f}, p-value={p_value:.4f} "
              f"{'normalny' if p_value > 0.05 else '✗ nienormalny'}")

    print(f"\n4b. Test Homogeniczności Wariancji (Levene) dla '{metric}':")
    print("-" * 60)

    group_data = [df[df[group_col] == g][metric].values for g in groups]
    stat, p_value = levene(*group_data)

    print(f"Statystyka: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Wniosek: {'✓ Wariancje jednorodne' if p_value > 0.05 else '✗ Wariancje niejednorodne'}")

    return normality_results, (stat, p_value)


def anova_analysis(df, metric='f1 score', group_col='architecture'):
    print("\n" + "=" * 80)
    print("KROK 5: ANALIZA WARIANCJI (ANOVA)")
    print("=" * 80)

    groups = df[group_col].unique()
    group_data = [df[df[group_col] == g][metric].values for g in groups]

    # ANOVA
    f_stat, p_value = f_oneway(*group_data)

    print(f"\nTest ANOVA dla '{metric}' względem '{group_col}':")
    print(f"F-statystyka: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"\n Istnieją ISTOTNE różnice między grupami (p < 0.05)")
    else:
        print(f"\n Brak istotnych różnic między grupami (p ≥ 0.05)")

    return f_stat, p_value

def pairwise_t_tests(df, metric='f1 score', group_col='architecture',
                     correction='bonferroni'):
    """
    Przeprowadza testy t-Studenta dla wszystkich par grup.
    Stosuje korektę Bonferroniego dla wielokrotnych porównań.
    """
    print("\n" + "=" * 80)
    print("KROK 6: TESTY POST-HOC (pairwise t-tests)")
    print("=" * 80)

    groups = df[group_col].unique()
    pairs = list(combinations(groups, 2))

    results = []

    print(f"\nLiczba porównań: {len(pairs)}")
    if correction == 'bonferroni':
        alpha_corrected = 0.05 / len(pairs)
        print(f"Korekta Bonferroniego: α = 0.05 / {len(pairs)} = {alpha_corrected:.6f}")

    print("\nWyniki testów t-Studenta:")
    print("-" * 80)

    for group1, group2 in pairs:
        data1 = df[df[group_col] == group1][metric].values
        data2 = df[df[group_col] == group2][metric].values

        # Test t-Studenta (niezależne próby)
        t_stat, p_value = ttest_ind(data1, data2)

        # Wielkość efektu (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

        is_significant = p_value < (alpha_corrected if correction == 'bonferroni' else 0.05)

        results.append({
            'Group 1': group1,
            'Group 2': group2,
            'Mean 1': np.mean(data1),
            'Mean 2': np.mean(data2),
            'Diff': np.mean(data1) - np.mean(data2),
            't-statistic': t_stat,
            'p-value': p_value,
            'Cohen\'s d': cohens_d,
            'Significant': is_significant
        })

        print(f"{group1:15s} vs {group2:15s}: "
              f"t={t_stat:7.3f}, p={p_value:.6f}, d={cohens_d:6.3f} "
              f"{'***' if is_significant else ''}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')

    return results_df

def rank_combinations(agg_df, metric='weighted_f1'):
    """
    Tworzy ranking kombinacji i identyfikuje najlepsze/najgorsze.
    """
    print("\n" + "=" * 80)
    print("KROK 7: RANKING KOMBINACJI")
    print("=" * 80)

    ranked = agg_df.sort_values(metric, ascending=False).copy()
    ranked['rank'] = range(1, len(ranked) + 1)

    print(f"\n TOP 5 najlepszych kombinacji (według {metric}):")
    print("-" * 80)
    print(ranked.head(5).to_string(index=False))

    print(f"\n TOP 5 najsłabszych kombinacji:")
    print("-" * 80)
    print(ranked.tail(5).to_string(index=False))

    # Analiza różnic
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]

    print(f"\n PORÓWNANIE:")
    print(f"Najlepsza:  {best['architecture']:15s} + {best['encoder']:10s} = {best[metric]:.4f}")
    print(f"Najsłabsza: {worst['architecture']:15s} + {worst['encoder']:10s} = {worst[metric]:.4f}")
    print(
        f"Różnica:    {best[metric] - worst[metric]:.4f} ({((best[metric] - worst[metric]) / worst[metric] * 100):.2f}%)")

    return ranked

def analyze_factors(df, agg_df):
    """
    Analizuje wpływ architektury i encodera osobno.
    """
    print("\n" + "=" * 80)
    print("KROK 8: ANALIZA CZYNNIKÓW")
    print("=" * 80)

    # Wpływ architektury
    print("\n8a. Średnie wyniki według ARCHITEKTURY:")
    print("-" * 60)
    arch_means = agg_df.groupby('architecture')['weighted_f1'].agg(['mean', 'std', 'count'])
    arch_means = arch_means.sort_values('mean', ascending=False)
    print(arch_means)

    # Wpływ encodera
    print("\n8b. Średnie wyniki według ENCODERA:")
    print("-" * 60)
    enc_means = agg_df.groupby('encoder')['weighted_f1'].agg(['mean', 'std', 'count'])
    enc_means = enc_means.sort_values('mean', ascending=False)
    print(enc_means)

    # Wykres
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot dla architektury
    df_melted = df.copy()
    df_melted = df_melted.merge(
        agg_df[['architecture', 'encoder', 'weighted_f1']],
        on=['architecture', 'encoder']
    )

    sns.boxplot(data=df, x='architecture', y='f1 score', ax=axes[0])
    axes[0].set_title('Rozkład F1-score według Architektury')
    axes[0].set_xlabel('Architektura')
    axes[0].set_ylabel('F1 Score')
    axes[0].tick_params(axis='x', rotation=45)

    # Boxplot dla encodera
    sns.boxplot(data=df, x='encoder', y='f1 score', ax=axes[1])
    axes[1].set_title('Rozkład F1-score według Encodera')
    axes[1].set_xlabel('Encoder')
    axes[1].set_ylabel('F1 Score')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('factor_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Zapisano wykres: factor_analysis.png")

    return arch_means, enc_means

def analyze_per_class(df):
    """
    Analizuje wydajność dla każdej klasy sentymentu osobno.
    """
    print("\n" + "=" * 80)
    print("KROK 9: ANALIZA PER KLASA")
    print("=" * 80)

    for emotion in df['emotion'].unique():
        print(f"\n📌 Klasa: {emotion.upper()}")
        print("-" * 60)

        subset = df[df['emotion'] == emotion].copy()

        # Najlepsze kombinacje dla tej klasy
        best_combo = subset.loc[subset['f1 score'].idxmax()]

        print(f"Najlepsza kombinacja:")
        print(f"  {best_combo['architecture']} + {best_combo['encoder']}")
        print(f"  F1: {best_combo['f1 score']:.4f}")
        print(f"  Precision: {best_combo['precision']:.4f}")
        print(f"  Recall: {best_combo['recall']:.4f}")

        # Statystyki
        print(f"\nStatystyki dla klasy '{emotion}':")
        print(f"  Średnia F1: {subset['f1 score'].mean():.4f}")
        print(f"  Odch. std:  {subset['f1 score'].std():.4f}")
        print(f"  Min F1:     {subset['f1 score'].min():.4f}")
        print(f"  Max F1:     {subset['f1 score'].max():.4f}")


def create_comprehensive_visualization(df, agg_df):
    """
    Tworzy kompleksowy dashboard wizualizacji.
    """
    print("\n" + "=" * 80)
    print("KROK 10: TWORZENIE WIZUALIZACJI")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Heatmapa wyników
    ax1 = fig.add_subplot(gs[0, :2])
    pivot = agg_df.pivot(index='architecture', columns='encoder', values='weighted_f1')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax1,
                cbar_kws={'label': 'Weighted F1'})
    ax1.set_title('Heatmapa Weighted F1: Architektura × Encoder', fontsize=12, fontweight='bold')

    # 2. Ranking
    ax2 = fig.add_subplot(gs[0, 2])
    top_5 = agg_df.head(5).copy()
    top_5['combination'] = top_5['architecture'] + '\n' + top_5['encoder']
    ax2.barh(range(len(top_5)), top_5['weighted_f1'], color=sns.color_palette('viridis', len(top_5)))
    ax2.set_yticks(range(len(top_5)))
    ax2.set_yticklabels(top_5['combination'], fontsize=8)
    ax2.set_xlabel('Weighted F1')
    ax2.set_title('TOP 5 Kombinacji', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()

    # 3. Porównanie metryk
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_comparison = agg_df.groupby('architecture')[['mean_precision', 'mean_recall', 'weighted_f1']].mean()
    metrics_comparison.plot(kind='bar', ax=ax3, rot=45)
    ax3.set_title('Średnie Metryki według Architektury', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Wartość')
    ax3.legend(['Precision', 'Recall', 'F1'], fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Encoder comparison
    ax4 = fig.add_subplot(gs[1, 1])
    encoder_comparison = agg_df.groupby('encoder')[['mean_precision', 'mean_recall', 'weighted_f1']].mean()
    encoder_comparison.plot(kind='bar', ax=ax4, rot=45)
    ax4.set_title('Średnie Metryki według Encodera', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Wartość')
    ax4.legend(['Precision', 'Recall', 'F1'], fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # 5. Per class performance
    ax5 = fig.add_subplot(gs[1, 2])
    per_class = df.groupby('emotion')['f1 score'].mean().sort_values(ascending=False)
    colors = ['#2ecc71' if x > 0.8 else '#f39c12' if x > 0.6 else '#e74c3c' for x in per_class.values]
    ax5.bar(range(len(per_class)), per_class.values, color=colors)
    ax5.set_xticks(range(len(per_class)))
    ax5.set_xticklabels(per_class.index, rotation=45)
    ax5.set_ylabel('Średni F1 Score')
    ax5.set_title('Wydajność per Klasa', fontsize=10, fontweight='bold')
    ax5.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Distribution of F1 scores
    ax6 = fig.add_subplot(gs[2, :])
    for arch in df['architecture'].unique():
        subset = df[df['architecture'] == arch]['f1 score']
        ax6.hist(subset, alpha=0.5, label=arch, bins=20)
    ax6.set_xlabel('F1 Score')
    ax6.set_ylabel('Częstość')
    ax6.set_title('Rozkład F1 Scores według Architektury', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    print("\n✓ Zapisano dashboard: comprehensive_dashboard.png")
    plt.close()

def generate_summary(agg_df, arch_means, enc_means, correlation_matrix):
    """
    Generuje podsumowanie i wnioski z analizy.
    """
    print("\n" + "=" * 80)
    print("KROK 11: PODSUMOWANIE I WNIOSKI")
    print("=" * 80)

    best = agg_df.iloc[0]
    worst = agg_df.iloc[-1]

    print("\n KLUCZOWE WNIOSKI:")
    print("-" * 80)

    print(f"\n1. NAJLEPSZA KOMBINACJA:")
    print(f"   Architektura: {best['architecture']}")
    print(f"   Encoder: {best['encoder']}")
    print(f"   Weighted F1: {best['weighted_f1']:.4f}")

    print(f"\n2. NAJSŁABSZA KOMBINACJA:")
    print(f"   Architektura: {worst['architecture']}")
    print(f"   Encoder: {worst['encoder']}")
    print(f"   Weighted F1: {worst['weighted_f1']:.4f}")

    print(f"\n3. NAJLEPSZA ARCHITEKTURA:")
    best_arch = arch_means.index[0]
    print(f"   {best_arch} (średnia F1: {arch_means.loc[best_arch, 'mean']:.4f})")

    print(f"\n4. NAJLEPSZY ENCODER:")
    best_enc = enc_means.index[0]
    print(f"   {best_enc} (średnia F1: {enc_means.loc[best_enc, 'mean']:.4f})")

    print(f"\n5. KORELACJE:")
    print(f"   Precision-Recall: {correlation_matrix.loc['precision', 'recall']:.3f}")
    print(f"   Precision-F1: {correlation_matrix.loc['precision', 'f1 score']:.3f}")
    print(f"   Recall-F1: {correlation_matrix.loc['recall', 'f1 score']:.3f}")

    with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PODSUMOWANIE ANALIZY MULTI-TASK LEARNING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Najlepsza kombinacja: {best['architecture']} + {best['encoder']} (F1: {best['weighted_f1']:.4f})\n")
        f.write(
            f"Najsłabsza kombinacja: {worst['architecture']} + {worst['encoder']} (F1: {worst['weighted_f1']:.4f})\n")
        f.write(f"\nRóżnica: {best['weighted_f1'] - worst['weighted_f1']:.4f}\n")
        f.write(f"\nPełny ranking:\n")
        f.write(agg_df.to_string(index=False))

    print("\n✓ Zapisano podsumowanie: analysis_summary.txt")


def run_complete_analysis(data_dict=None, csv_path=None):
    """
    Uruchamia pełną analizę krok po kroku.
    """
    print("\n" + "=" * 80)
    print("ROZPOCZĘCIE KOMPLEKSOWEJ ANALIZY DATA MINING")
    print("=" * 80)

    # Krok 1: Wczytanie danych
    df = load_and_prepare_data(data_dict, csv_path)

    # Krok 2: Agregacja metryk
    agg_df = aggregate_metrics(df)

    # Krok 3: Analiza korelacji
    corr_matrix = correlation_analysis(df)

    # Krok 4: Testy założeń
    normality_arch, homogeneity_arch = test_assumptions(df, 'f1 score', 'architecture')
    normality_enc, homogeneity_enc = test_assumptions(df, 'f1 score', 'encoder')

    # Krok 5: ANOVA
    anova_arch = anova_analysis(df, 'f1 score', 'architecture')
    anova_enc = anova_analysis(df, 'f1 score', 'encoder')

    # Krok 6: Testy post-hoc
    pairwise_arch = pairwise_t_tests(df, 'f1 score', 'architecture')
    pairwise_enc = pairwise_t_tests(df, 'f1 score', 'encoder')

    # Krok 7: Ranking
    ranked = rank_combinations(agg_df)

    # Krok 8: Analiza czynników
    arch_means, enc_means = analyze_factors(df, agg_df)

    # Krok 9: Analiza per klasa
    analyze_per_class(df)

    # Krok 10: Wizualizacje
    create_comprehensive_visualization(df, agg_df)

    # Krok 11: Podsumowanie
    generate_summary(agg_df, arch_means, enc_means, corr_matrix)

    print("\n" + "=" * 80)
    print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print("=" * 80)
    print("\nWygenerowane pliki:")
    print("  • correlation_matrix.png")
    print("  • factor_analysis.png")
    print("  • comprehensive_dashboard.png")
    print("  • analysis_summary.txt")

    return {
        'dataframe': df,
        'aggregated': agg_df,
        'ranked': ranked,
        'correlations': corr_matrix,
        'architecture_means': arch_means,
        'encoder_means': enc_means,
        'pairwise_arch': pairwise_arch,
        'pairwise_enc': pairwise_enc
    }

if __name__ == "__main__":

    results = run_complete_analysis(csv_path='C:/Users/Julia xD/DataspellProjects/meisd_project/pipeline/EMOTIA/EMOTIA-ML/outputs_multitask/EmotionClassification.csv')
