"""Task4 module for predicting local authorities based on postcode data."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

class Task4:
    """Class for training and predicting local authorities."""

    def __init__(self, train):
        """
        Initialize Task4 instance.

        Parameters:
        - train (pd.DataFrame): The training data.
        """
        self.df = train
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
        X = self.df.drop(columns=['localAuthority'])
        y = self.df['localAuthority']
        return X, y

    def _train_pipeline(self, num_columns=['easting', 'northing']):
        """
        Train the machine learning pipeline.

        Parameters:
        - num_columns (list): Numeric features for processing.
        """
        X, y = self.prepare_data()

        num_transformer = ColumnTransformer([
            ('simpleImputer', SimpleImputer(strategy='mean'), num_columns)
        ])

        self.pipeline = Pipeline([
            ('num_transformer', num_transformer),
            ('knn', KNeighborsClassifier(n_neighbors=3))
        ])

        self.pipeline.fit(X, y)

    def predict(self, test):
        """
        Make predictions using the trained model.

        Parameters:
        - test (pd.DataFrame): Data for making predictions.

        Returns:
        - predictions (np.ndarray): Predicted local authorities.
        """
        predictions = self.pipeline.predict(test)
        return predictions

if __name__ == "__main__":
    # Read training data
    train = pd.read_csv('flood_tool/resources/postcodes_labelled.csv')
    # Initialize and train the model
    model_instance = Task4(train)
    # Read test data
    test = pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv')
    # Make predictions
    predictions = model_instance.predict(test)
    print(predictions)
