import os
from src.heartstrokeprediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.heartstrokeprediction.entity.config_entity import DataTransformationConfig

class DataTransformation: 
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        # Load the dataset
        data = pd.read_csv(self.config.data_path)

        # Drop the 'id' column from the dataset if it exists
        if 'id' in data.columns:
            data.drop(columns=['id'], inplace=True)
            logger.info("Dropped 'id' column from the dataset")

        # Calculate the mean value of the 'bmi' column and fill NaN values with the mean
        mean_value = data['bmi'].mean()
        data['bmi'].fillna(mean_value, inplace=True)

        # Apply label encoding to categorical columns
        label_encoder = LabelEncoder()
        data['gender'] = label_encoder.fit_transform(data['gender'])
        data['ever_married'] = label_encoder.fit_transform(data['ever_married'])
        data['work_type'] = label_encoder.fit_transform(data['work_type'])
        data['Residence_type'] = label_encoder.fit_transform(data['Residence_type'])
        data['smoking_status'] = label_encoder.fit_transform(data['smoking_status'])
        
        logger.info("Applied label encoding to categorical columns")

        # Split the data into features and target (assuming the target is the last column)
        X = data.drop(columns=['stroke'])
        y = data['stroke']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Save the transformed data to CSV files
        train_transformed = pd.concat([X_train, y_train], axis=1)
        test_transformed = pd.concat([X_test, y_test], axis=1)
        
        train_transformed.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_transformed.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted and transformed data into training and test sets")
        logger.info(f"Train shape: {train_transformed.shape}")
        logger.info(f"Test shape: {test_transformed.shape}")

        print(f"Train shape: {train_transformed.shape}")
        print(f"Test shape: {test_transformed.shape}")
        print(f"Train target column 'stroke' has {y_train.isnull().sum()} NaN values.")
        print(f"Test target column 'stroke' has {y_test.isnull().sum()} NaN values.")