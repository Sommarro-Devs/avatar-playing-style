# avatar-playing-style

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

Preferably use the requirements.txt file eg.
pip install -r requirements.txt or if using conda: conda install -r requirements.txt

## Downloads
------------
Make sure to have Python3 downloaded, along with needed packages listed above.

Get the Wyscout data from: https://figshare.com/collections/Soccer_match_event_dataset/4415000/2 

The following data sets from Wyscout are needed: "events.json", "matches.json", "players.json" and "teams.json".

Place the downloaded Wyscout data in a folder named: `Wyscout`, placed two levels above the Python code (see below).

Also download Excel-sheet `Gameweek_38.xlsx` from XXXXXXXX and place at one level above the Python code (see below).

## Running Instructions
------------



## Project Organization
------------

    ├── README.md                               <- The top-level README for running this project.
    |
    ├── Wyscout                                 <- Wyscout data folder.
    │   │
    │   ├── players.json
    │   │
    │   ├── teams.json  
    │   │
    │   ├── events            
    │   │   ├── events_England.json
    │   │   ├── events_France.json
    │   │   ├── events_Germany.json
    │   │   ├── events_Italy.json
    │   │   └── events_Spain.json
    │   │
    │   └── matches            
    │       ├── matches_England.json
    │       ├── matches_France.json
    │       ├── matches_Germany.json
    │       ├── matches_Italy.json
    │       └── matches_Spain.json
    │
    └──Player_rating_Project                    <- Main folder for this project.
        |
        │── Gameweek_38.xlsx                    <- Excel with validation data from Whoscored to compare with.
        │
        │── Json_files                          <- Folder where created json-files are stored.
        │
        └── Python_Code                         <- Source code for this project.
            |
            |── create_events_df_eu.py
            |── create_KPI_dataframe_EDIT.py
            |── create_KPI_dataframe.py
            |── FCPython.py
            |── fitting_functions.py
            |── GW_38_Ratings_evaluation.py
            |── GW_38_Ratings.py
            |── KPI_functions.py
            |── minutes_played.py
            |── the_match_ranking.py
            |── validation_vs_WhoScored.py
            └── xG_model_evaluation.py

--------

By: Jakob Edberger Persson and Emil Danielsson, 2022

