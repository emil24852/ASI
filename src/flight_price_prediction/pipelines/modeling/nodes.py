import pandas as pd
from flaml.automl import AutoML
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def scale_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    X_train = df_train.drop('price', axis=1).fillna(0)
    X_test  = df_test.drop('price', axis=1).fillna(0)
    y_train = df_train['price']
    y_test  = df_test['price']

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled.fillna(0), X_test_scaled.fillna(0), y_train, y_test


def extract_feature_columns(X_train: pd.DataFrame) -> list[str]:
    return X_train.columns.tolist()


def automl_train(X_train: pd.DataFrame, y_train: pd.Series, time_budget: int = 300) -> object:
    automl = AutoML()
    automl_settings = {
        "time_budget":    time_budget,
        "metric":         "rmse",
        "task":           "regression",
        "log_file_name":  "flaml.log"
    }

    automl.fit(
        X_train=X_train.to_numpy(),
        y_train=y_train.to_numpy(),
        **automl_settings
    )

    return automl.model


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    preds = model.predict(X_test)
    return pd.DataFrame([{
        "model": model.__class__.__name__,
        "mae":   mean_absolute_error(y_test, preds),
        "rmse":  np.sqrt(mean_squared_error(y_test, preds)),
        "r2":    r2_score(y_test, preds)
    }])
