"""
Konwersja danych z formatu emotion1/2/3 do one-hot encoding
Użyj PRZED treningiem classifiera
"""

import pandas as pd
import numpy as np

def convert_to_onehot(input_csv, output_csv):
    """
    Konwertuje CSV z formatem:
      emotion1, intensity1, emotion2, intensity2, emotion3, intensity3

    Na format:
      emotion__joy, intensity__joy, emotion__sadness, intensity__sadness, ...
    """

    df = pd.read_csv(input_csv, encoding='utf-8')

    # Znajdź kolumnę tekstową
    text_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ['utterance', 'text', 'augmented', 'message'])]
    text_col = text_cols[0] if text_cols else 'text'

    # Zbierz wszystkie unikalne emocje
    all_emotions = set()
    for col in ['emotion1', 'emotion2', 'emotion3']:
        if col in df.columns:
            emotions = df[col].dropna().astype(str).str.lower().str.strip()
            all_emotions.update(emotions.unique())

    # Usuń puste wartości
    all_emotions = sorted([e for e in all_emotions if e and e != 'nan'])

    print(f"Found emotions: {all_emotions}")

    # Nowy DataFrame
    new_rows = []

    for idx, row in df.iterrows():
        new_row = {
            text_col: row[text_col],
            'sentiment': row.get('sentiment', 'neutral')
        }

        # Zbierz emocje z tego wiersza
        present_emotions = {}
        for i in [1, 2, 3]:
            e_col = f'emotion{i}'
            i_col = f'intensity{i}'

            if e_col in df.columns and pd.notna(row.get(e_col)):
                emotion = str(row[e_col]).lower().strip()
                if emotion and emotion != 'nan':
                    intensity = row.get(i_col, 2)
                    try:
                        intensity = int(float(intensity))
                    except:
                        intensity = 2
                    present_emotions[emotion] = intensity

        # One-hot encoding
        for emotion in all_emotions:
            new_row[f'emotion__{emotion}'] = 1 if emotion in present_emotions else 0
            new_row[f'intensity__{emotion}'] = present_emotions.get(emotion, 0)

        # Zachowaj inne kolumny jeśli istnieją
        for col in ['original', 'quality', 'mode']:
            if col in df.columns:
                new_row[col] = row[col]

        new_rows.append(new_row)

    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"\nConverted {len(df_new)} samples")
    print(f"   Emotions: {len(all_emotions)}")
    print(f"   Saved to: {output_csv}")

    return df_new

# Przykładowe użycie
if __name__ == "__main__":
    input_file = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/EMOTIA/EMOTIA-DA/outputs22112025/MEISD_balanced_expanded.csv" #ESConv_balanced_expanded_2D.csv"  # Twój augmentowany plik
    output_file = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/EMOTIA/EMOTIA-DA/outputs22112025/multilabel_augmented_onehot_11222025.csv"

    df_converted = convert_to_onehot(input_file, output_file)

    # Sprawdź format
    print("\nColumn names:")
    print(df_converted.columns.tolist())

    print("\nSample row:")
    print(df_converted.iloc[0].to_dict())