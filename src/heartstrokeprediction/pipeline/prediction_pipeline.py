import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load the model and scaler (ensure these are saved correctly during training)
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.scaler = joblib.load(Path('artifacts/model_trainer/scaler.pkl'))  # Load the scaler

    def predict(self, data):
        """
        Function to predict stroke based on the given input data.
        :param data: The input data for prediction.
        :return: Prediction result (0 or 1).
        """
        # Assuming 'data' is a numpy array with the shape (1, number of features)
        # Scale the input data (you must have the same number of features and scaling applied as during training)
        data_scaled = self.scaler.transform(data)
        
        # Make prediction using the loaded model
        prediction = self.model.predict(data_scaled)
        
        # Return the prediction (0 or 1)
        return prediction
