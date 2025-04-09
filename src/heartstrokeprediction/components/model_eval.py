import os
from src.heartstrokeprediction import logger
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.heartstrokeprediction.entity.config_entity import ModelEvaluationConfig    
from pathlib import Path
from src.heartstrokeprediction.constants import *
from src.heartstrokeprediction.utils.common import read_yaml, create_directories,save_json


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/kk1455217/Heart-Stroke-Prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="kk1455217"
os.environ["MLFLOW_TRACKING_PASSWORD"]="9c05d7cb76433d44cb358f7431105d416ba10413"
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """Calculate evaluation metrics: RMSE, MAE, R2"""
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """Evaluate model and log results into MLflow"""
        # Load the test data
        test_data = pd.read_csv(self.config.test_data_path)

        # Load the model and scaler
        model = joblib.load(self.config.model_path)
        scaler = joblib.load(self.config.pkl_path)

        # Separate features (X) and target (y)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Scale the test features using the saved scaler
        test_x_scaled = scaler.transform(test_x)

        # Start an MLflow run
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            # Make predictions on the scaled test data
            predicted_qualities = model.predict(test_x_scaled)

            # Evaluate the model and calculate metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save the metrics as a JSON file
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters to MLflow
            mlflow.log_params(self.config.all_params)

            # Log metrics to MLflow
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # If not using a file store, register the model in the Model Registry
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="HeartStrokePredictionModel")
            else:
                mlflow.sklearn.log_model(model, "model")

            print(f"Evaluation metrics saved and model logged in MLflow!")