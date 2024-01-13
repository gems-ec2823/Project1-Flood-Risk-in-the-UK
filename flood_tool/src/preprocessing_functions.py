def add_postcode_features(df):
    listed_postcodes = df['postcode'].values.tolist()
    postcode_areas, postcode_districts, postcode_sectors, postcode_units = [], [], [], []

    for elem in listed_postcodes:

        if len(elem) == 7:
            postcode_area = elem[:2]
            postcode_district = elem[:3]
            postcode_sector = elem[:3]+elem[4:5]
            postcode_unit = elem[:3]+elem[4:7]

        if len(elem) == 6:
            postcode_area = elem[:1]
            postcode_district = elem[:2]
            postcode_sector = elem[:2]+elem[3:4]
            postcode_unit = elem[:2]+elem[3:6]
        
        if len(elem) == 8:
            postcode_area = elem[:2]
            postcode_district = elem[:4]
            postcode_sector = elem[:4]+elem[5:6]
            postcode_unit = elem[:4]+elem[5:8]

        postcode_areas.append(postcode_area)
        postcode_districts.append(postcode_district)
        postcode_sectors.append(postcode_sector)
        postcode_units.append(postcode_unit)
    df['postcode_area'] = postcode_areas
    df['postcode_district'] = postcode_districts
    df['postcode_sector'] = postcode_sectors
    df['postcode_unit'] = postcode_units
    return df

def filter_by_percentile(df):
    lower = df['medianPrice'].quantile(0.1)
    upper = df['medianPrice'].quantile(0.9)
    return df[(df['medianPrice'] >= lower) & (df['medianPrice'] <= upper)]

def merging_dataframes(df1, df2, left_on, right_on, how='left'):
    merged = df1.merge(df2, left_on=left_on, right_on=right_on, how=how)
    return merged

def modify_postcodeSector(postcode):
    postcode = postcode.replace(' ','')
    postcode = postcode.upper()
    return postcode

def simplify_postcode(postcode):
    parts = postcode.split()
    if len(parts) > 1:
        return parts[0] + ' ' + parts[1][0]
    return parts[0]


def standardize_postcode(postcode):
    # Remove spaces and convert to uppercase
    standardized_postcode = postcode.replace(" ", "").upper()
    
    # Check if it is in standard format
    if (len(standardized_postcode) == 6 or len(standardized_postcode) == 7) and standardized_postcode[2].isdigit():
        # Insert a space after the 3rd character for length 6
        if len(standardized_postcode) == 6:
            standardized_postcode = standardized_postcode[:3] + " " + standardized_postcode[3:]
            return standardized_postcode
        else:
            # Insert a space after the 4th character for length 7
            standardized_postcode = standardized_postcode[:4] + " " + standardized_postcode[4:]
            return standardized_postcode
    else:
        # Insert a space after the 4th character for length 5
        standardized_postcode = standardized_postcode[:2] + " " + standardized_postcode[2:]
        return standardized_postcode

def incomes_preprocessing(incomes):
    columns = incomes.iloc[0,:].values
    incomes.iloc[1:,:]
    incomes.columns = columns
    incomes = incomes.iloc[1:,:]
    incomes['Total annual income (£)'] = incomes['Total annual income (£)'].str.replace(',', '').astype(float)
    incomes['Total annual income (£)'].groupby(incomes['Local authority name']).mean()
    res = incomes['Total annual income (£)'].groupby(incomes['Local authority name']).mean()
    res = res.reset_index()
    return res
