"""Example module in template package."""

import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from .geo import *  # noqa: F401, F403
from .src.task1 import Task1
from .src.task2 import Task2
from .src.task3 import Task3
from .src.task4 import Task4
from .src.task2_easy import Task2_easy

# from geo import *  # noqa: F401, F403
# from src.task1 import Task1
# from src.task2 import Task2
# from src.task3 import Task3
# from src.task4 import Task4

from .src.preprocessing_functions import standardize_postcode, add_postcode_features, modify_postcodeSector, merging_dataframes

script_path = os.path.abspath(__file__)
flood_tool_directory = os.path.dirname(script_path)

__all__ = [
    "Tool",
    "_data_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
]

_data_dir = os.path.join(os.path.dirname(__file__), "resources")

# Methods doing their respective prediction tasks
flood_class_from_postcode_methods = {
    "zero_risk": "All zero risk",
    "flood_class_from_postcode_tree": "Decision Tree Pipeline for Flood Class from Postcode"
}

flood_class_from_location_methods = {
    "zero_risk": "All zero risk",
    "flood_class_from_locations_tree": "Decision Tree Pipeline for Flood Class from Locations"
}

historic_flooding_methods = {
    "all_false": "All False",
    "historic_flooding_rf": "Random Forest Pipeline for Historic Flooding"
}

house_price_methods = {
    "all_england_median": "All England Median",
    "house_price_rf_filter": "Random Forest Pipeline for House Price",
    # "house_price_rf_easy": "Random Forest Pipeline for House Price without filter"
}

local_authority_methods = {
    "do_nothing": "Do Nothing",
    "local_authority_knn": "KNN for Local Authority"
}


class Tool:
    """Class to interact with a postcode database file and train models."""

    def __init__(self, unlabelled_unit_data="", labelled_unit_data="",
                 sector_data="", district_data="", additional_data={}):
        """
        Initialize Tool instance.

        Parameters:
        - unlabelled_unit_data (str, optional): Filename of a .csv file containing geographic location data for postcodes.
        - labelled_unit_data (str, optional): Filename of a .csv containing class labels for specific postcodes.
        - sector_data (str, optional): Filename of a .csv file containing information on households by postcode sector.
        - district_data (str, optional): Filename of a .csv file containing information on households by postcode district.
        - additional_data (dict, optional): Dictionary containing additional .csv files containing additional information on households.
        """

        if unlabelled_unit_data == "":
            unlabelled_unit_data = os.path.join(_data_dir, 'postcodes_unlabelled.csv')

        if labelled_unit_data == "":
            labelled_unit_data = os.path.join(_data_dir, 'postcodes_labelled.csv')

        if sector_data == "":
            sector_data = os.path.join(_data_dir, 'households_per_sector.csv')

        self.postcode_unlabelled = pd.read_csv(unlabelled_unit_data)
        self.postcode_labelled = pd.read_csv(labelled_unit_data)

        db = pd.concat([self.postcode_unlabelled, self.postcode_labelled])
        db = db[['postcode', 'easting', 'northing', 'soilType', 'elevation']]
        self.db = db.drop_duplicates(subset=['postcode'])
        self.db = self.db.set_index('postcode')

        self.models_trained = {}

    def train(self, models=[]):
        """Train models using a labelled set of samples.

        Parameters:
        - models (list): Sequence of model keys to train.

        Examples:
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.train(fcp_methods[0])
        >>> classes = tool.predict_flood_class_from_postcode(['M34 7QL'], fcp_methods[0])
        """

        for model in models:
            if model in self.models_trained:
                print(f"Model '{model}' is already trained. Skipping training.")
            else:
                if model == 'flood_class_from_postcode_tree' or model == 'flood_class_from_locations_tree':
                    model = 'flood_class_from_postcode_tree'
                    task1_model = Task1(self.postcode_labelled)
                    self.models_trained[model] = task1_model
                elif model == 'house_price_rf_filter':
                    task2_model = Task2(self.postcode_labelled)
                    self.models_trained[model] = task2_model
                elif model == 'house_price_rf_easy':
                    task2_model = Task2_easy(self.postcode_labelled)
                    self.models_trained[model] = task2_model
                elif model == 'historic_flooding_rf':
                    task3_model = Task3(self.postcode_labelled)
                    self.models_trained[model] = task3_model
                else:
                    print(f"Model '{model}' is not recognized and will not be trained.")

                task4_model = Task4(self.postcode_labelled)
                self.models_trained["local_authority_knn"] = task4_model

    def lookup_easting_northing(self, postcodes, dtype=np.float64):
        """Get a DataFrame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of OSGB36 easting and northing,
            indexed by the input postcodes. Invalid postcodes (i.e., those
            not in the input unlabelled postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['M34 7QL'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                    easting  northing
        postcode
        M34 7QL    393470   394371
        >>> results = tool.lookup_easting_northing(['M34 7QL', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                    easting  northing
        postcode
        M34 7QL  393470.0  394371.0
        AB1 2PQ       NaN       NaN
        """

        frame = self.db.copy()
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing"]]
    
    def lookup_lat_long(self, postcodes):
        """Get a Pandas DataFrame containing GPS latitude and longitude
        information for a collection of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e., those not in
            the input unlabelled postcodes file) return as NaN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL'])
                    latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        """
        frame = self.db.copy()
        frame = frame.reindex(postcodes)

        frame = self.lookup_easting_northing(postcodes=postcodes)
        frame['latitude'], frame['longitude'] = get_gps_lat_long_from_easting_northing(frame['easting'], frame['northing'])

        return frame.loc[:, ['longitude', 'latitude']]


    def predict_flood_class_from_postcode(self, postcodes, method="zero_risk"):
        """
        Generate series predicting flood probability classification
        for a collection of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """
        if type(postcodes) == list:
            postcodes = pd.Series(postcodes)
        postcodes = postcodes.apply(standardize_postcode)

        if method == "zero_risk":
            return pd.Series(
                data=np.ones(len(postcodes), int),
                index=np.asarray(postcodes),
                name="riskLabel",
            )

        if method == 'flood_class_from_postcode_tree':
            X = self.db.loc[postcodes].reset_index(drop=False)
            task1_model = self.models_trained[method]
            y_pred = task1_model.predict(X)
            return pd.Series(data=y_pred, index=np.asarray(postcodes), name="riskLabel")
  
    def predict_flood_class_from_OSGB36_location(
        self, eastings, northings, method="zero_risk"
    ):
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        """

        if method == "zero_risk":
            return pd.Series(
                data=np.ones(len(eastings), int),
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
            )
        
        elif method == 'flood_class_from_locations_tree':
            method = 'flood_class_from_postcode_tree'
            loc = pd.DataFrame({'easting': eastings, 'northing': northings})
            X = self.db.reset_index()
            merged_df = pd.merge(X, loc, on=['easting', 'northing'])
            merged_df.dropna(inplace=True)
            merged_df.drop_duplicates(inplace=True)
            task1_model = self.models_trained[method]
            y_pred = task1_model.predict(merged_df)
            return pd.Series(
                data=np.ones(len(eastings), int),
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
            )


    def predict_flood_class_from_WGS84_locations(
        self, longitudes, latitudes, method="zero_risk"
    ):
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : int (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if method == "zero_risk":
            return pd.Series(
                data=np.ones(len(longitudes), int),
                index=[(lng, lat) for lng, lat in zip(longitudes, latitudes)],
                name="riskLabel",
            )
        if method == 'flood_class_from_location_tree':
            eastings, northings  = get_easting_northing_from_gps_lat_long(latitudes, longitudes, dtype=int)
            return self.predict_flood_class_from_OSGB36_location(eastings, northings, method)


    def predict_median_house_price(
        self, postcodes, method="all_england_median"
    ):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """

        if type(postcodes) == list:
            postcodes = pd.Series(postcodes)

        postcodes = postcodes.apply(standardize_postcode)

        if method == "all_england_median":
            return pd.Series(
                data=np.full(len(postcodes), 245000.0),
                index=np.asarray(postcodes),
                name="medianPrice",
            )
        
        if method == "house_price_rf_filter":
            X = self.db.loc[postcodes].reset_index(drop=False)
            task2_model = self.models_trained[method]
            y_pred = task2_model.predict(X)
            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="medianPrice",
                )
        
        if method == "house_price_rf_easy":
            X = self.db.loc[postcodes].reset_index(drop=False)
            task2_easy_model = self.models_trained[method]
            y_pred = task2_easy_model.predict(X)
            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="medianPrice",
                )

    def predict_local_authority(
        self, eastings, northings, method="do_nothing"
    ):
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            postcodes, and indexed by postcodes.
        """

        if method == "do_nothing":
                return pd.Series(
                    data=np.full(len(eastings), np.nan),
                    index=[(est, nth) for est, nth in zip(eastings, northings)],
                    name="localAuthority",
                )
        if method == "local_authority_knn":
                X = pd.DataFrame({'easting': eastings, 'northing': northings})
                task4_model = self.models_trained[method]
                y_pred = task4_model.predict(X)
                return pd.Series(data=y_pred,
                    index=[(est, nth) for est, nth in zip(eastings, northings)],
                    name="localAuthority",
                    )
                
    def predict_historic_flooding(
        self, postcodes, method="all_false"
    ):
        """
        Generate series predicting local authorities in m for a sequence
        of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        """ 

        if type(postcodes) == list:
            postcodes = pd.Series(postcodes)

        postcodes = postcodes.apply(standardize_postcode)

        if method == "all_false":
            return pd.Series(
                data=np.full(len(postcodes), False),
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )
        
        if method == "historic_flooding_rf":
            X = self.db.loc[postcodes].reset_index(drop=False)
            task3_model = self.models_trained[method]
            y_pred = task3_model.predict(X)
            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )

    def predict_total_value(self, postal_data):
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcodesectors

        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        df = self.db.copy()
        indexes = df.index

        df = df.reset_index(drop=False)
        df = add_postcode_features(df)

        inter = list(set(indexes) & set(postal_data))

        if inter != []:
            print(self.models_trained.keys())
            price = self.predict_median_house_price(postal_data, method="house_price_rf_filter")
            file_path = flood_tool_directory + '/resources/sector_data.csv'
            sector_data = pd.read_csv(file_path)
            sector_data['postcodeSector'] = sector_data['postcodeSector'].apply(modify_postcodeSector)
            df = merging_dataframes(df, sector_data, left_on='postcode_sector', right_on='postcodeSector', how='left')

            post = price.index.tolist()
            filtered_df = df[df['postcode'].isin(post)]
            # print(filtered_df)
            filtered_df['price'] = price.values

            filtered_df['result'] = filtered_df['price'] * filtered_df['households']

            res = filtered_df['result']
            return pd.Series(
                data=res,
                index=np.asarray(postal_data),
                name="h",
            )


    def predict_annual_flood_risk(self, postcodes, risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        risk_labels = risk_labels or self.get_flood_class(postcodes)

        raise NotImplementedError

tool = Tool()
# fcp_methods = list(house_price_methods.keys())
# print(fcp_methods)
# tool.train(fcp_methods)


# methods = list(local_authority_methods.keys())  # doctest: +SKIP
# tool.train(methods)  
# y_pred = tool.predict_local_authority([390978, 396607, 427859], [403269, 298083, 432937], methods[1])
# print(y_pred)

# methods = list(house_price_methods.keys())  # doctest: +SKIP
# print(methods)
# tool.train(methods)  
# y_pred = tool.predict_median_house_price(['M34 7QL', 'OL4 3NQ', 'B36 8TE', 'NE16 3AT', 'WS10 8D'], methods[1])
# print(y_pred)

# methods = list(historic_flooding_methods.keys())  # doctest: +SKIP
# print(methods)
# tool.train(methods)
# y_pred = tool.predict_historic_flooding(['M34 7QL', 'OL4 3NQ', 'B36 8TE', 'NE16 3AT', 'WS10 8DE', 'OL9 7NS', 'WV13 2LR', 'LS12 1LZ', 'SK15 1TS', 'TS17 9NN', 'LE6 5TE', 'DL7 7TB'], methods[1])
# print(y_pred)

# methods = list(flood_class_from_location_methods.keys())  # doctest: +SKIP
# tool.train(methods)  
# print(methods)

# eas = [409215,419672,411674]
# north = [416819,560517,288499]

# lat, long = get_gps_lat_long_from_easting_northing(eas, north)
    
# y_pred = tool.predict_flood_class_from_WGS84_locations(lat, long, methods[1])
# print(y_pred)




# df = '/Users/azp123/Desktop/EDSML/Projects/ads-deluge-dart/flood_tool/resources/postcodes_labelled.csv'
# df =  pd.read_csv(df)

# methods = list(house_price_methods.keys())  # doctest: +SKIP
# print(methods)
# # tool.train(methods)  

# postal_data = df.postcode.tolist()
# # print(postal_data)

# tool.train(methods)
# y = tool.predict_total_value(postal_data)
# print(y)