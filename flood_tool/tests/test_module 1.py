"""Test Module."""

import pytest
import sys
import pandas as pd
import numpy as np
sys.path.append('..')
from flood_tool import *
tool = Tool()



def test_get_easting_northing_from_gps_lat_long():
    """Check if we could get the easting and northing when enter lattitude and longitude"""

    gps_coords = [55.5, -1.54]
    expected_easting_northing = (np.array([429157]), np.array([623009]))
    easting, northing = get_easting_northing_from_gps_lat_long(gps_coords[0], gps_coords[1], dtype=int)
    assert np.array_equal(easting, expected_easting_northing[0])
    assert np.array_equal(northing, expected_easting_northing[1])

def test_lookup_easting_northing():
    """Check if we could get the easting and northing when enter postcode"""

    data = tool.lookup_easting_northing(['M34 7QL'])

    assert np.isclose(data.loc['M34 7QL', 'easting'], 393470, atol=1e-3)
    assert np.isclose(data.loc['M34 7QL', 'northing'], 394371, atol=1e-3)

    invalid_data = tool.lookup_easting_northing(['AB1 2CD'])
    assert invalid_data.isna().all().all()


def test_lookup_lat_long():
    """Check if we could get the lattitude and longitude when enter postcode"""
    data = tool.lookup_lat_long(["M34 7QL"])

    assert np.isclose(data.iloc[0].latitude, 53.4461, rtol=1.0e-3).all()
    assert np.isclose(data.iloc[0].longitude, -2.0997, rtol=1.0e-3).all()
    
def test_get_gps_lat_long_from_easting_northing():
    """Check if we could get the lattitude and longitude when enter leasting and northing"""
    
    easting_northing = (np.array([429157]), np.array([623009]))
    expected_gps = [55.5, -1.54]
    longitude, latitude = get_gps_lat_long_from_easting_northing(easting_northing[0],easting_northing[1])
    assert np.isclose(longitude, expected_gps[0], rtol=1.0e-3).all()
    assert np.isclose(latitude, expected_gps[1], rtol=1.0e-3).all()
    
# def test_predict_flood_class_from_postcode():
    
#     data = tool.predict_flood_class_from_postcode(["OL9 7NS"], method='flood_class_from_postcode_tree')
#     expected_class = 1
    
#     assert np.array_equal(data.values[0], expected_class = 1)

# def test_predict_median_house_price():
    
#     data = tool.predict_median_house_price(["OL9 7NS"], method='house_price_rf_filter')
#     expected_price = 0
    
#     assert data.values[0]>=0
    