import streamlit as st
import pandas as pd
import pickle
import os

from src.flight_price_prediction.pipelines.data_preparation.nodes import (
    clean_data,
    duration_to_minutes,
    encode_features,
)

MODEL_PATH   = "data/06_models/best_model_automl.pkl"
COLUMNS_PATH = "data/06_models/model_columns.pkl"
DATA_PATH    = "data/01_raw/data1.csv"

@st.cache_data(show_spinner=False)
def load_reference():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {DATA_PATH}")
    if not os.path.isfile(COLUMNS_PATH):
        raise FileNotFoundError("Brakuje pliku z listą kolumn modelu (model_columns.pkl)")

    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)

    unique_vals = {
        "airline":          sorted(df["airline"].dropna().unique()),
        "source_city":      sorted(df["source_city"].dropna().unique()),
        "destination_city": sorted(df["destination_city"].dropna().unique()),
        "departure_time":   sorted(df["departure_time"].dropna().unique()),
        "arrival_time":     sorted(df["arrival_time"].dropna().unique()),
        "class":            sorted(df["class"].dropna().unique()),
        "stops":            sorted(df["stops"].dropna().unique()),
    }

    with open(COLUMNS_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    return unique_vals, feature_columns

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def main():
    st.title("✈️ Flight Price Prediction")
    st.write("Wybierz parametry lotu w panelu po lewej i kliknij \"Oblicz cenę\".")

    try:
        unique_vals, feature_columns = load_reference()
    except Exception as e:
        st.error(str(e))
        return

    try:
        model = load_model()
    except Exception as e:
        st.error(str(e))
        return

    st.sidebar.header("Parametry lotu")
    airline      = st.sidebar.selectbox("Linia lotnicza", unique_vals["airline"])
    source_city  = st.sidebar.selectbox("Miasto wylotu", unique_vals["source_city"])
    dest_city    = st.sidebar.selectbox("Miasto docelowe", unique_vals["destination_city"])
    dep_time     = st.sidebar.selectbox("Czas wylotu", unique_vals["departure_time"])
    arr_time     = st.sidebar.selectbox("Czas przylotu", unique_vals["arrival_time"])
    travel_class = st.sidebar.selectbox("Klasa", unique_vals["class"])
    stops        = st.sidebar.selectbox("Liczba przystanków", unique_vals["stops"])
    duration     = st.sidebar.number_input("Czas lotu (h.mm)", value=2.0, step=0.1)
    days_left    = st.sidebar.number_input("Dni do wylotu", min_value=0, value=1, step=1)

    if st.sidebar.button("Oblicz cenę"):
        row = pd.DataFrame([{
            "airline":          airline,
            "source_city":      source_city,
            "destination_city": dest_city,
            "departure_time":   dep_time,
            "arrival_time":     arr_time,
            "class":            travel_class,
            "stops":            stops,
            "duration":         duration,
            "days_left":        days_left,
        }])

        row = clean_data(row)
        row = duration_to_minutes(row)
        row_enc = encode_features(row)

        X = row_enc.reindex(columns=feature_columns, fill_value=0).fillna(0)

        try:
            price = model.predict(X)[0]
        except Exception as e:
            st.error(f"Błąd predykcji: {e}")
            return

        st.subheader("Szacowana cena biletu")
        st.write(f"### {price:.2f} PLN")


if __name__ == "__main__":
    main()
