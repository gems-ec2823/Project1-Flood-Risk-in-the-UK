# Flood Risk Prediction tool

## Deadlines
-  **Code: 12pm GMT Friday 24th November 2023**
-  **Presentation and one-page report: 4pm GMT Friday 24th November 2023**

You should update this document in the course of your work to reflect the scope and abilities of your project, as well as to provide appropriate instuctions to potential users (and markers) of your code.

### Key Requirements

Your project must provide the following:

 1. at least one analysis method to estimate a number of attributes for unlabelled postcodes extrapolated from sample data which is provided to you:
    - Flood risk from rivers & seas (on a 10 point scale).
    - Flood risk from surface water (on a 10 point scale).
    - Median house price.
 2. at least one analysis method to estimate the Local Authority & flood risks of arbitrary locations. 
 3. a process to find the rainfall and water level near a given postcode from provided rainfall, river and tide level data, or by looking online.
 4. visualization and analysis tools for the postcode, rainfall, river & tide data provided to you, ideally in a way which will identify potential areas at immediate risk of flooding by combining the data you have been given.
 
 Your code should have installation instructions and basic documentation, either as docstrings for functions & class methods, a full manual or both.

![London postcode density](images/LondonPostcodeDensity.png)
![England Flood Risk](images/EnglandFloodRisk.png)
![UK soil types](images/UKSoilTypes.png)

This README file *should be updated* over the course of your group's work to represent the scope and abilities of your project.

### Assessment

 - Your code will be assessed for its speed (both at training and prediction) & predictive accuracy.
 - Your code should include tests of its functionality.
 - Additional marks will be awarded for maintaining good code quality and for a clean, well-organised repository. You should consider the kind of code you would be happy to take over from another group of students.

 ### Software Installation Guide

***Software requirements***
`Python` 
`pip`
`git`

***Install this repository by running***
```bash
git clone https://github.com/ese-msc-2023/ads-deluge-dart.git
```
***Install the requirements***

*With conda*

go to `ads-deluge-dart` directory and then run the commands:

```bash
conda env create -f environment.yml
conda activate deluge
```

*With pip*

from `ads-deluge-dart` directory run

```bash
pip install .
```

Your installation is now done and the flood tool is ready to be used

The Flood-Tool package is a comprehensive tool designed for predictive modeling of flood risk based on geographical data and flood-related information. Leveraging machine learning algorithms, this package utilizes datasets containing postcodes' geographical coordinates and additional features to train robust models for predicting flood risk levels.

### User instructions

If you want to import this package:

```bash
import flood_tool as ft
```

To use the Flood_tool package, start by creating an instance of the Tool class, which will serve as your central interface for flood risk assessment. Then, train your flood risk assessment models using the provided training methods. For example, you can train a decision tree model for predicting flood risk from postcodes. Feel free to adjust the class to incorporate additional models and integrate them seamlessly within the class structure.

Suppose you possess two CSV datasets one labeled for flood prediction and another without a target variable. Ensure that both datasets share common columns, namely 'postcode,' 'easting,' 'northing,' 'elevation,' and 'soilType.' Additionally, ensure that both datasets are located within the resources folder.

```bash
tool = ft.Tool().tool.train(['flood_class_from_postcode_tree', 'flood_class_from_locations_tree', 'historic_flooding_rf', 'house_price_rf_filter'])
```

***Flood risk label prediction***


```bash
tool.predict_flood_class_from_postcode(pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv').postcode.to_list())
```
This will give you a flood risk label on a scale from 1 to 10. The likelyhood of a flood can be assumed to be:

M34 7QL     1<br>
OL4 3NQ     1<br>
B36 8TE     1<br>
NE16 3AT    1<br>
WS10 8DE    1<br>
           ..<br>
NN9 7TY     1<br>
HU6 7YG     1<br>
LS12 1DY    1<br>
DN4 6TZ     1<br>
S31 9BD     1<br>
Name: riskLabel, Length: 10000, dtype: int64<br>


***median houseprice prediction***

```bash
tool.predict_median_house_price(pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv').postcode.to_list())
```

Out[18]: <br>
M34 7QL     245000.0<br>
OL4 3NQ     245000.0<br>
B36 8TE     245000.0<br>
NE16 3AT    245000.0<br>
WS10 8DE    245000.0<br>
              ...<br>
NN9 7TY     245000.0<br>
HU6 7YG     245000.0<br>
LS12 1DY    245000.0<br>
DN4 6TZ     245000.0<br>
S31 9BD     245000.0<br>
Name: medianPrice, Length: 10000, dtype: float64<br>


***Historic flooding prediction***
To predict the *historic flooding* navigate to `ads-deluge-dart` and run

```bash
tool.predict_historic_flooding(pd.read_csv('flood_tool/resources/postcodes_unlabelled.csv').postcode.to_list())
```

Out[15]: <br>
M34 7QL     False<br>
OL4 3NQ     False<br>
B36 8TE     False<br>
NE16 3AT    False<br>
WS10 8DE    False<br>
            ...<br>
NN9 7TY     False<br>
HU6 7YG     False<br>
LS12 1DY    False<br>
DN4 6TZ     False<br>
S31 9BD     False<br>
Name: historicallyFlooded, Length: 10000, dtype: bool<br>


***Note***: You can do the same for all tests using this line:

To view the performance scores on your custom train-test split, navigate to the project folder and execute the following command: python -m scoring. This command will provide you with a comprehensive summary.

```bash
python -m scoring
```


***Visualisation***

You can find the visualization part in the repository inside flood_tool folder and it is called "Vizualisation_Grapgh2.ipynb"




*Add report*

The code includes a basic [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be built by running

```bash
python -m sphinx docs html
```

then viewing the generated `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual PDF can be generated by running

```bash
python -m sphinx  -b latex docs latex
```

Then follow the instructions to process the `FloodTool.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to check its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```bash
python -m pytest --doctest-modules flood_tool
```

### Reading list (this can be updated as you go)

 - [A guide to coordinate systems in Great Britain](https://webarchive.nationalarchives.gov.uk/20081023180830/http://www.ordnancesurvey.co.uk/oswebsite/gps/information/coordinatesystemsinfo/guidecontents/index.html)

 - [Information on postcode validity](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/283357/ILRSpecification2013_14Appendix_C_Dec2012_v1.pdf)
