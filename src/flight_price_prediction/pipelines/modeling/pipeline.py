from kedro.pipeline import node, pipeline, Pipeline
from .nodes import scale_data, automl_train, evaluate_model, extract_feature_columns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scale_data,
            inputs=["train_data", "test_data"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="scale_data_node"
        ),
        node(
            func=extract_feature_columns,
            inputs="X_train",
            outputs="model_columns",
            name="extract_feature_columns_node"
        ),
        node(
            func=automl_train,
            inputs=["X_train", "y_train"],
            outputs="best_model",
            name="automl_train_node"
        ),
        node(
            func=evaluate_model,
            inputs=["best_model", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node"
        ),
    ])
