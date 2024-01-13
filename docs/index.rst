#####################
The Team Dart Flood Tool
#####################

This package implements a flood risk prediction and visualization tool.

Installation Instructions
-------------------------

Overview
^^^^^^^^
This guide provides step-by-step instructions for installing the necessary components of the project. Please follow the instructions in the order presented for a smooth setup experience.

Prerequisites
^^^^^^^^^^^^^
Before proceeding with the installation, ensure you have Python installed on your system. This project is compatible with Python 3.x versions.

Installing the Main Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open your terminal or command prompt.

2. Navigate to the directory where the project's `setup.py` file is located.

3. Run the following command:

.. code-block:: bash

    python setup.py install

This will install the main package along with any dependencies specified in the `setup.py` file.

Setting Up the Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For users utilizing Conda for environment management, the project provides an `environment.yml` file. This file contains all the necessary packages and their versions to create a Conda environment. Follow these steps:

1. Open your terminal or Anaconda Prompt.

2. Ensure you have Conda installed and updated to the latest version.

3. Run the following command in the directory containing `environment.yml`:

.. code-block:: bash

    conda env create -f environment.yml

4. Activate the newly created environment using:

.. code-block:: bash

    conda activate [environment_name]

Replace `[environment_name]` with the name specified in the `environment.yml` file.

Quick Usage guide
-----------------

Overview
^^^^^^^^
This guide provides quick instructions on how to use the flood risk prediction tool and the visualisation tool included in the `ads-deluge-dart` project.

Flood Risk Prediction Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^

Purpose
"""""""
The Flood Risk Prediction Tool is designed to assess and predict flood risks in specified geographical areas.

Getting Started
"""""""""""""""
1. **Navigation**: First, navigate to the `ads-deluge-dart` directory in your terminal or command prompt.

2. **Execution**: Run the model with the following command:

   .. code-block:: bash

       python tool.py

Optional Arguments
""""""""""""""""""
You can enhance the model's functionality with up to three optional arguments:

.. code-block:: bash

    python tool.py

- **unlabelled_unit_data**: Path to a `.csv` file with geographic location data for postcodes. Refer to `flood_tool/resources/postcodes_unlabelled.csv` for the required format.
- **labelled_unit_data**: Path to a `.csv` file with labelled geographic location data for postcodes. Refer to `flood_tool/resources/postcodes_labelled.csv` for the required format.
- **sector_data**: Path to a `.csv` file with households information by postcode. Refer to `flood_tool/resources/sector.csv` for the required format.


Default File Paths
""""""""""""""""""
If no arguments are provided, the tool uses default files:

- **postcode_labelled**: `flood_tool/resources/postcodes_labelled.csv`
- **postcodes_unlabelled**: `flood_tool/resources/postcodes_unlabelled.csv`
- **households_file**: `flood_tool/resources/sector.csv`

Output
""""""
Upon completion, the model outputs risk assessments, labels, and median property price predictions for the input locations. These can be found in the `flood_tool/predictions/` directory.

Visualisation Tool
^^^^^^^^^^^^^^^^^^

Purpose
"""""""
The Visualisation Tool helps in graphically representing the flood risk data, making it easier to interpret and analyze.

Using the Tool
""""""""""""""
1. **Navigate** to the `ads-deluge-dart` directory in your terminal or command prompt.

2. **Launch** the visualisation tool by executing:

   .. code-block:: bash

       python tool.py

Features
""""""""
- **Data Visualization**: This tool provides visual representation of flood risk predictions, including heatmaps, charts, and graphs.

- **Interactive Interface**: It offers an interactive interface for exploring different data points and geographic locations.

Note
""""
For detailed documentation on using these tools, including advanced configurations and troubleshooting, please refer to the full user manual or visit the project's online documentation page.


Further Documentation
---------------------


.. toctree::
   :maxdepth: 2

   models
   coordinates
   visualization


Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:

