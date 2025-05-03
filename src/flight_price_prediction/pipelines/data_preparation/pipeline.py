from kedro.pipeline import node, pipeline, Pipeline
from .nodes import clean_data, duration_to_minutes, encode_features, train_test_split

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs="raw_data",
            outputs="cleaned_data",
            name="clean_data_node"
        ),
        node(
            func=duration_to_minutes,
            inputs="cleaned_data",
            outputs="data_duration",
            name="duration_node"
        ),
        node(
            func=encode_features,
            inputs="data_duration",
            outputs="feateng_data",
            name="encode_node"
        ),
        node(
            func=train_test_split,
            inputs="feateng_data",
            outputs=["train_data", "test_data"],
            name="split_node"
        ),
    ])
