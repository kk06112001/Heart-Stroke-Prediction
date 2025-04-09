import pandas as pd
import os
from src.heartstrokeprediction import logger
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import StandardScaler
from src.heartstrokeprediction.entity.config_entity import ModelTrainerConfig
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load the training and testing data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split features (X) and target (y) for both train and test datasets
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Standard scaling (feature normalization) - fit the scaler on the training set
        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)  # Fit and transform the training data
        test_x_scaled = scaler.transform(test_x)  # Only transform the test data

        # Initialize and train the model (using Linear Regression for regression task)
        model = LogisticRegression(random_state=42)  # Replace with LinearRegression() if regression
        model.fit(train_x_scaled, train_y)

        # Predict on both train and test sets
        train_predictions = model.predict(train_x_scaled)
        test_predictions = model.predict(test_x_scaled)

        # Save the trained model and scaler
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.pkl"))  # Save the scaler

        print(f"Model saved to {os.path.join(self.config.root_dir, self.config.model_name)}")
        print(f"Scaler saved to {os.path.join(self.config.root_dir, 'scaler.pkl')}")