# avatar-playing-style
The developed model aims to classify an individual football player’s playing style solely based on his or hers event data collected from football matches he or she has participated in. Done in collaboration with Football Analytics AB as a masters thesis in engineering physics spring term of 2022, Uppsala University.

For further backround and results regardning the master thesis, feel free to look at the reports folder.

## Content:
1. [Introduction](#Intro)
2. [Setup](#Setup)
3. [Project Organization](#Project)
4. [Notebooks](#Notebooks)

------------
## Introduction <a class="anchor" id="Intro"></a>

Project has been carried out at Uppsala university in collaoration with Football Analytics Sweden AB,as a masters thesis in engineering physics, VT2022.

------------
## Setup <a class="anchor" id="Setup"></a>
Make sure you have the following packages downloaded in your virtual environment:
- `pandas`
- `numpy`
- `matplotlib`
- `mplsoccer`
- `scikit_learn`
- `scipy`
- `statsmodels`
- `tabulate`
- `Pillow`

Preferably use the requirements.txt file to download them by either typing

pip install -r requirements.txt

or if you are using Anaconda:

conda install --file requirements.txt

## Project Organization <a class="anchor" id="Project"></a>
------------

    ├── README.md    <- The top-level README for running this project.
    |
    ├── data         <- Folder with match event data, kpi-data and validation data.
    |
    ├── figures_used <- Folder with figures used in the project, mainly PlayMaker logo.
    |
    ├── reports      <- Folder with written material related to the project such as hypothesis, the final master thesis report and appendix.
    |
    └──  lib         <- Root folder for the code of the project with python notebooks and the module folder.
        ├── position_detection.ipynb
        ├── model_v1.ipynb
        ├── model_v2.ipynb
        ├── model_v2_viz.ipynb
        │
        └── modules   <- Folder with module files used in the project.                       
            ├── config.py
            ├── data_processing_lib.py
            ├── helpers_lib.py
            ├── models_lib.py
            ├── plot_styles.py
            ├── radar_class.py
            ├── validation_lib.py
            └── viz_lib.py


## Notebooks <a class="anchor" id="Notebooks"></a>

### position_detection.ipynb
Notebook with validation and vizualisations for the position detection model.

### model_v1.ipynb
Notebook with validation for Model v.1.

### model_v2.ipynb
Notebook with validation for Model v.2.

### model_v2_viz.ipynb
Notebook with GUIs for spider vizualizations of the Model v.2 results.

--------

By: Jakob Edberger Persson and Emil Danielsson, 2022

