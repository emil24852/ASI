import os
import io
import pickle
import pandas as pd
from azure.storage.blob import BlobServiceClient

AZ_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_MODELS= os.getenv("BLOB_MODELS_CONTAINER", "models")
BLOB_DATA = os.getenv("BLOB_DATA_CONTAINER", "data")

service_client = BlobServiceClient.from_connection_string(AZ_CONN)

def _download_blob(container: str, blob_name: str) -> bytes:
    container_client = service_client.get_container_client(container)
    downloader = container_client.download_blob(blob_name)
    return downloader.readall()

def load_blob_df(blob_name: str) -> pd.DataFrame:
    raw = _download_blob(BLOB_DATA, blob_name)
    return pd.read_csv(io.BytesIO(raw))

def load_blob_pickle(blob_name: str):
    raw = _download_blob(BLOB_MODELS, blob_name)
    return pickle.load(io.BytesIO(raw))
