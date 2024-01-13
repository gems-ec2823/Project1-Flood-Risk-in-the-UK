import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class OrdinalTransformer(TransformerMixin, BaseEstimator):
    """Custom transformer to convert categorical features to ordinal integers based on a provided mapping."""

    def __init__(self, category_mapping, unknown='ignore'):
        """
        Initialize the OrdinalTransformer.

        Parameters:
        - category_mapping (dict): Mapping of column names to lists of categories in the desired ordinal order.
        - unknown (str): Strategy for handling unknown categories. Options: 'ignore' (default), 'use_max'.
        """
        self.category_mapping = category_mapping
        self.unknown = unknown
        self.category_dicts = {col: {cat: idx for idx, cat in enumerate(categories)} for col, categories in category_mapping.items()}

    def fit(self, X=None, y=None):
        """
        Fit the transformer.

        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target data (ignored).

        Returns:
        - self: This instance.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform input data by converting categorical features to ordinal integers.

        Parameters:
        - X (pd.DataFrame): Input data.

        Returns:
        - X_transformed (pd.DataFrame): Transformed data.
        """
        X = pd.DataFrame(X, columns=['soilType'])
        X_transformed = X.copy()
        for col, categories in self.category_mapping.items():
            X_transformed[col] = X[col].apply(lambda x: self.category_dicts[col].get(x, self.handle_unknown(col, x)))
        return X_transformed

    def handle_unknown(self, column, value):
        """
        Handle unknown categories based on the specified strategy.

        Parameters:
        - column (str): Name of the column with the unknown category.
        - value: The unknown category.

        Returns:
        - int: Ordinal value for the unknown category.
        """
        if self.unknown == 'ignore':
            return value
        elif self.unknown == 'use_max':
            return max(self.category_dicts[column].values()) + 1
        else:
            raise ValueError(f"Unknown handling mode '{self.unknown}' not supported.")
