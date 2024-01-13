"""Task3 module for predicting historical flood occurrences based on postcode data."""

import os
import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from .transformers import OrdinalTransformer
from .preprocessing_functions import add_postcode_features

# Ignore warnings
warnings.filterwarnings('ignore')
# Mapping for soil types
soilType_mapping = {'soilType': ['Luvisols', 'Cambisols', 'Arenosols', 'Leptosols', 'Podsols', 'Planosols', 'Stagnosols', 'Gleysols', 'Histosols', 'Unsurveyed/Urban']}

script_path = os.path.abspath(__file__)
src_directory = os.path.dirname(script_path)
flood_tool_directory = os.path.dirname(src_directory)


class Task3:
    """Class for training and predicting historical flood occurrences."""

    def __init__(self, train, soilType_mapping=soilType_mapping):
        """
        Initialize Task3 instance.

        Parameters:
        - data (pd.DataFrame): The input data.
        - soilType_mapping (dict): Mapping for soil types.
        """
        self.df = train
        self.soilType_mapping = soilType_mapping
        self._load_data()
        self._train_pipeline()

    def _load_data(self):
        """Drop NaN values from the DataFrame."""
        self.df.dropna(inplace=True)

    def prepare_data(self):
        """
        Prepare data for training.

        Returns:
        - X (pd.DataFrame): Features for training.
        - y (pd.Series): Target variable for training.
        """
        self.df = add_postcode_features(self.df)

        X = self.df.drop(columns=['riskLabel', 'medianPrice', 'historicallyFlooded', 'localAuthority'])
        y = self.df['historicallyFlooded']
        return X, y

    def _train_pipeline(self, num_features=['easting', 'northing', 'elevation'],
                        ord_features=['soilType'], cat_features=['postcode_district']):
        """
        Train the machine learning pipeline.

        Parameters:
        - num_features (list): Numeric features for processing.
        - ord_features (list): Ordinal features for processing.
        - cat_features (list): Categorical features for processing.
        """
        X, y = self.prepare_data()

        preprocessor = ColumnTransformer([
            ('num_transformer', make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()), num_features),
            ('cat_transformer', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                              OneHotEncoder(sparse_output=False, handle_unknown='ignore')), cat_features),
            ('ord_transformer', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                              OrdinalTransformer(self.soilType_mapping)), ord_features)
        ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        self.pipeline.fit(X, y)

    def predict(self, test):
        """
        Make predictions using the trained model.

        Parameters:
        - test (pd.DataFrame): Data for making predictions.

        Returns:
        - predictions (np.ndarray): Predicted historical flood occurrences.
        """
        test = add_postcode_features(test)
        predictions = self.pipeline.predict(test)
        return predictions


if __name__ == "__main__":
    # Read training data
    train = pd.read_csv('flood_tool/resources/postcodes_labelled.csv')
    # Initialize and train the model
    model_instance = Task3(train, soilType_mapping)
    # Read test data
    test = pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv')
    # Make predictions
    predictions = model_instance.predict(test)
    print(predictions)

