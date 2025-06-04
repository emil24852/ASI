import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Flight Price Prediction", layout="wide")

@st.cache_data(show_spinner=False)
def load_model(path="best_model_automl.pkl"):
    """
    Wczytuje picklowany model (pipeline lub XGBRegressor)
    oczekujący numerycznych wektorów cech (dummy‐zmienne dla 4 kolumn).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Plik modelu nie został znaleziony: {path}\n"
            "Upewnij się, że best_model_automl.pkl jest w katalogu, "
            "z którego odpala się `streamlit run app.py`."
        )
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "brakujący moduł"
        raise RuntimeError(
            f"Brakuje modułu: {missing}\n"
            "Ten model został prawdopodobnie zapisany z użyciem FLAML-a lub innego pakietu.\n"
            f"Aby to naprawić, wykonaj:\n    pip install {missing}"
        )
    except Exception as e:
        msg = str(e).lower()
        if "libomp.dylib" in msg or "libgomp.so" in msg or "xgboost" in msg:
            raise RuntimeError(
                "Nie udało się załadować bibliotek XGBoost lub brakuje OpenMP (libomp).\n"
                "Na macOS wykonaj:\n    brew install libomp\n"
                "Następnie spróbuj ponownie `streamlit run app.py`."
            )
        else:
            raise RuntimeError(f"Nie udało się wczytać modelu z {path}\nSzczegóły: {e}")
    return model

@st.cache_data(show_spinner=False)
def load_reference_data(path="Clean_Dataset.csv"):
    """
    Wczytuje Clean_Dataset.csv (tylko po to, by zebrać unikalne wartości
    w czterech kolumnach kategorycznych oraz zbudować pełną listę kolumn dummy).
    Zwraca:
      - unique_vals: słownik list unikalnych wartości do dropdownów,
      - feature_columns: lista nazw wszystkich dummy‐kolumn, jakie model widział przy treningu.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Plik danych referencyjnych nie znaleziono: {path}\n"
            "Upewnij się, że Clean_Dataset.csv jest w tym samym folderze co app.py."
        )

    df = pd.read_csv(path)
    # Usuń 'Unnamed: 0', jeśli się pojawiło jako indeks:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 1. Sprawdźmy, czy mamy dokładnie te cztery kolumny w CSV:
    needed = ["source_city", "destination_city", "arrival_time", "class"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Brak kolumny '{col}' w pliku Clean_Dataset.csv")

    # 2. Pobranie unikalnych wartości (do interfejsu):
    unique_vals = {
        "source_city": sorted(df["source_city"].dropna().unique()),
        "destination_city": sorted(df["destination_city"].dropna().unique()),
        "arrival_time": sorted(df["arrival_time"].dropna().unique()),
        "class": sorted(df["class"].dropna().unique()),
    }

    # 3. Zbudujmy pełny zestaw dummy‐zmiennych dla tych czterech kolumn:
    df_cats = df[needed]
    df_dummies_full = pd.get_dummies(df_cats, prefix_sep="=")

    # 4. feature_columns to lista wszystkich kolumn z df_dummies_full:
    feature_columns = list(df_dummies_full.columns)

    return unique_vals, feature_columns

def main():
    st.title("✈️ Flight Price Prediction")
    st.write(
        """
        Wypełnij formularz po lewej, podając:
        - **Miasto wylotu** (`source_city`)  
        - **Miasto docelowe** (`destination_city`)  
        - **Czas przylotu** (`arrival_time`)  
        - **Klasa** (`class`)  

        Aplikacja zakoduje te cztery wartości jako one‐hot (dummy) 
        i użyje wytrenowanego modelu, aby oszacować **najniższą cenę** biletu (w PLN).
        """
    )

    # ===== 1. Wczytaj wytrenowany model =====
    try:
        model = load_model("best_model_automl.pkl")
    except Exception as e:
        st.error(str(e))
        return

    # ===== 2. Wczytaj dane referencyjne, by mieć dropdowny i pełną listę dummy‐kolumn =====
    try:
        unique_vals, feature_columns = load_reference_data("Clean_Dataset.csv")
    except Exception as e:
        st.error(str(e))
        return

    # ===== 3. Sidebar: formularz z dropdownami dla 4 pól =====
    st.sidebar.header("Parametry lotu")

    source_sel = st.sidebar.selectbox(
        "Miasto wylotu:",
        unique_vals["source_city"],
    )
    dest_sel = st.sidebar.selectbox(
        "Miasto docelowe:",
        unique_vals["destination_city"],
    )
    arrival_sel = st.sidebar.selectbox(
        "Czas przylotu:",
        unique_vals["arrival_time"],
    )
    class_sel = st.sidebar.selectbox(
        "Klasa:",
        unique_vals["class"],
    )

    # ===== 4. Po kliknięciu “Oblicz cenę” – zakodujone‐hot i predykcja =====
    if st.sidebar.button("Oblicz cenę"):
        # 4a. Walidacja: upewnij się, że wszystkie cztery wartości są wybrane
        if not all([source_sel, dest_sel, arrival_sel, class_sel]):
            st.warning("▶️ Proszę upewnić się, że wszystkie cztery pola zostały wybrane.")
            return

        # 4b. Stwórz jednolinijkowy DataFrame tylko z czterema kolumnami
        df_one = pd.DataFrame([{
            "source_city": source_sel,
            "destination_city": dest_sel,
            "arrival_time": arrival_sel,
            "class": class_sel
        }])

        # 4c. Zamień na dummy‐zmienne tylko cztery kolumny
        df_dum = pd.get_dummies(df_one, prefix_sep="=")

        # 4d. Dopasuj (reindex) powstały zbiór dummy do pełnej listy feature_columns,
        #     wypełniając brakujące kolumny zerami:
        df_dum_aligned = df_dum.reindex(columns=feature_columns, fill_value=0)

        # 4e. Sprawdź, czy wszystkie kolumny są typu int/float/bool
        bad_dtypes = [
            col for col, dt in df_dum_aligned.dtypes.items()
            if dt not in [int, float, bool, "int64", "float64", "bool"]
        ]
        if bad_dtypes:
            st.error(f"Błąd w zakodowanych danych: nieprawidłowe typy w kolumnach: {bad_dtypes}")
            return

        # 4f. Predykcja – przekazujemy wyłącznie macierz numeryczną dummy‐zmiennych
        try:
            price_pred = model.predict(df_dum_aligned)[0]
        except Exception as e:
            st.error(
                "⚠️ Błąd podczas predykcji. Możliwe przyczyny:\n"
                "- Model nie oczekuje wyłącznie one‐hot encoding czterech cech,\n"
                "- Pipeline (jeśli jest w pickle’u) wymaga innych transformacji.\n\n"
                f"Szczegóły: {e}"
            )
            return

        # 4g. Wyświetl podsumowanie i wynik
        st.subheader("Podsumowanie wprowadzonych parametrów")
        st.write(f"- **Miasto wylotu**: {source_sel}")
        st.write(f"- **Miasto docelowe**: {dest_sel}")
        st.write(f"- **Czas przylotu**: {arrival_sel}")
        st.write(f"- **Klasa**: {class_sel}")

        st.markdown("---")
        st.subheader("Przewidywana najniższa cena biletu")
        st.write(f"## {price_pred:.2f} PLN")

    else:
        st.info("Wybierz parametry po lewej i kliknij **Oblicz cenę**.")

if __name__ == "__main__":
    main()