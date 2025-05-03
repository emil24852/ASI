from kedro.pipeline import Pipeline
from flight_price_prediction.pipelines.data_preparation.pipeline import create_pipeline as create_data_pipeline
from flight_price_prediction.pipelines.modeling.pipeline import create_pipeline as create_modeling_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    dp = create_data_pipeline()
    mp = create_modeling_pipeline()
    return {
        'data_preparation': dp,
        'modeling': mp,
        '__default__': dp + mp,
        }
