import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Usunięcie kolumny indeksu 'Unnamed: 0'
    - Usunięcie kolumny 'flight' (kody lotów nie wnoszą cech)
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])
    return df


def duration_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konwersja kolumny 'duration' z formatu h.mm do całkowitej liczby minut:
    - część całkowita to godziny
    - część po przecinku*100 to minuty
    """
    hours = df['duration'].astype(int)
    mins = ((df['duration'] - hours) * 100).astype(int)
    df['duration_mins'] = hours * 60 + mins
    return df.drop(columns=['duration'])


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inżynieria cech:
    1. Mapowanie liczby przystanków ('stops') na wartości numeryczne
    2. One-hot encoding dla wybranych kolumn kategorycznych
    3. Wypełnienie wszelkich braków zerami
    """
    stops_map = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four_or_more': 4
    }
    df['stops'] = df['stops'].map(stops_map).fillna(0).astype(int)

    cat_cols = [
        'airline',
        'source_city',
        'destination_city',
        'departure_time',
        'arrival_time',
        'class'
    ]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df.fillna(0)


def train_test_split(df: pd.DataFrame, frac: float = 0.8):
    """
    Podział na zbiór treningowy i testowy:
    - losowe przetasowanie wierszy
    - frac określa ułamek na trening (reszta to test)
    """
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(frac * len(df_shuffled))

    return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]
