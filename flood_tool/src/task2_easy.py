"""Task2 easy module for predicting median house prices based on postcode data."""

import os
import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from .preprocessing_functions import add_postcode_features, filter_by_percentile, merging_dataframes, modify_postcodeSector

# Ignore warnings
warnings.filterwarnings('ignore')

script_path = os.path.abspath(__file__)
src_directory = os.path.dirname(script_path)
flood_tool_directory = os.path.dirname(src_directory)

class Task2_easy:
    """Class for training and predicting median house prices."""

    def __init__(self, train):
        """
        Initialize Task2 instance.

        Parameters:
        - train (pd.DataFrame): The training data.
        """
        self.df = train
        self._load_data()
        self._train_pipeline()

    def _load_data(self):
        """Remove specific row and drop NaN values from the DataFrame."""
        self.df.drop(8713, inplace=True)
        self.df.dropna(inplace=True)

    def prepare_data(self):
        """
        Prepare data for training.

        Returns:
        - X (pd.DataFrame): Features for training.
        - y (pd.Series): Target variable for training.
        """
        self.df = add_postcode_features(self.df)

        file_path = flood_tool_directory + '/resources/sector_data.csv'
        sector_data = pd.read_csv(file_path)
        sector_data['postcodeSector'] = sector_data['postcodeSector'].apply(modify_postcodeSector)
        self.df = merging_dataframes(self.df, sector_data, left_on='postcode_sector', right_on='postcodeSector', how='left')
        # self.df = self.df.groupby('postcode_district').apply(filter_by_percentile)

        X = self.df.drop(columns=['medianPrice'])
        y = self.df['medianPrice']
        return X, y

    def _train_pipeline(self, num_features=['easting', 'northing', 'elevation', 'households'],
                        cat_features=['postcode_district']):
        """
        Train the machine learning pipeline.

        Parameters:
        - num_features (list): Numeric features for processing.
        - cat_features (list): Categorical features for processing.
        """
        X, y = self.prepare_data()
        preprocessor = ColumnTransformer([
            ('num_transformer', make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()), num_features),
            ('cat_transformer', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                              OneHotEncoder(sparse_output=False, handle_unknown='ignore')), cat_features),
        ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('rf', RandomForestRegressor(random_state=42, n_estimators=100))
        ])

        self.pipeline.fit(X, y)

    def predict(self, test):
        """
        Make predictions using the trained model.

        Parameters:
        - test (pd.DataFrame): Data for making predictions.

        Returns:
        - predictions (np.ndarray): Predicted median house prices.
        """
        test = add_postcode_features(test)
        file_path = flood_tool_directory + '/resources/sector_data.csv'
        test = merging_dataframes(test, pd.read_csv(file_path), left_on='postcode_sector', right_on='postcodeSector', how='left')
        predictions = self.pipeline.predict(test)
        return predictions


if __name__ == "__main__":
    # Initialize and train the model
    train = pd.read_csv('flood_tool/resources/postcodes_labelled.csv')
    model_instance = Task2_easy(train)
    # Read test data
    test = pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv')
    # Make predictions
    predictions = model_instance.predict(test)
    print(predictions)