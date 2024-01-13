"""Task1 module for predicting flood risk labels based on postcode data."""

import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from .transformers import OrdinalTransformer
from .preprocessing_functions import add_postcode_features

# Ignore warnings
warnings.filterwarnings('ignore')

# Mapping for soil types
soilType_mapping = {
    'soilType': [
        'Luvisols', 'Cambisols', 'Arenosols', 'Leptosols', 'Podsols',
        'Planosols', 'Stagnosols', 'Gleysols', 'Histosols', 'Unsurveyed/Urban'
    ]
}


class Task1:
    """Class for training and predicting flood risk labels."""

    def __init__(self, train, soilType_mapping=soilType_mapping):
        """
        Initialize Task1 instance.

        Parameters:
        - train (pd.DataFrame): The training data.
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
        X = self.df.drop(columns=['riskLabel', 'localAuthority', 'medianPrice', 'historicallyFlooded', 'postcode'])
        y = self.df['riskLabel']
        return X, y

    def _train_pipeline(self, num_features=['easting', 'northing', 'elevation'], ord_features=['soilType']):
        """Train the machine learning pipeline.

        Parameters:
        - num_features (list): Numeric features for processing.
        - ord_features (list): Ordinal features for processing.
        """
        X, y = self.prepare_data()

        preprocessor = ColumnTransformer([
            ('num_transformer', make_pipeline(SimpleImputer(strategy='mean'),
                                              RobustScaler()), num_features),
            ('ord_transformer', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                              OrdinalTransformer(self.soilType_mapping)), ord_features)
        ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('rf', DecisionTreeRegressor(random_state=42, ccp_alpha=0.0, criterion='squared_error', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, splitter='best'))
        ])

        self.pipeline.fit(X, y)

    def predict(self, test_data):
        """
        Make predictions using the trained model.

        Parameters:
        - test_data (pd.DataFrame): Data for making predictions.

        Returns:
        - predictions (np.ndarray): Predicted flood risk labels.
        """
        predictions = self.pipeline.predict(test_data)
        predictions = np.ceil(predictions).astype(int)
        return predictions


if __name__ == "__main__":
    # Read training data
    train = pd.read_csv('flood_tool/resources/postcodes_labelled.csv')
    # Initialize and train the model
    model_instance = Task1(train, soilType_mapping)
    # Read test data
    test = pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv')
    # Make predictions
    predictions = model_instance.predict(test)
    print(predictions)
