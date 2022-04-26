#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mars 21 10:00:31 2022

@author: JakobEP & emildanielsson

Program description:
    
    XXXXXXXXXXXXXXXX
    
"""

#%%
# - Imports
"---------------------------------------------------------------------------"

# Basics
import pandas as pd

# Project module
import avatarplayingstyle_module.validation_lib as validate
from avatarplayingstyle_module.config import dict_playingStyle_indices, list_all_playingStyle_indices
from avatarplayingstyle_module.models_lib import create_PCA_scores, map_PCA_scores


#%%
# - Input variables
"---------------------------------------------------------------------------"

# Choose position to validate for
set_position = 'CB'

# Choose league to validate
league = 'PL'
league = 'Swe'
league = 'both'


#%%
# - Read data
"---------------------------------------------------------------------------"

# Read model KPI dataframe from PL "train" data
df_KPI_PL = pd.read_excel('../data/processed/model_kpis_PL21-22_2022-04-09.xlsx')

# Read model KPI dataframe from Allsvenskan (test data)
df_KPI_Swe = pd.read_excel('../data/processed/model_kpis_Swe21_2022-04-09.xlsx')


#%%
# - call to model
"---------------------------------------------------------------------------"

# get PCA-scores
dict_PCA_result = create_PCA_scores(df_KPI_PL, df_KPI_Swe)
df_result_PCA_PL = dict_PCA_result['result_train']
df_result_PCA_PL_excl = dict_PCA_result['result_excl_train']
df_result_PCA_Swe = dict_PCA_result['result_test']
df_result_PCA_Swe_excl = dict_PCA_result['result_excl_test']

# map scores to playingstyles
df_playing_styles_PL = map_PCA_scores(df_result_PCA_PL, df_result_PCA_PL_excl)
df_playing_styles_Swe = map_PCA_scores(df_result_PCA_Swe, df_result_PCA_Swe_excl)


#%%
# - Handle inputs
"---------------------------------------------------------------------------"
df_playing_style = pd.DataFrame()
if league == 'PL':
    df_playing_style = df_playing_styles_PL
elif league == 'Swe':
    df_playing_style = df_playing_styles_Swe
elif league == 'both':
    df_playing_style = pd.concat([df_playing_styles_PL, df_playing_styles_Swe])
else: 
    print("WRONG INPUT")


#%%
# - Get results in confusion matrix
"---------------------------------------------------------------------------"

# Replace to validation index formatting
df_playing_style.replace({"The Target": 1.1, "The Poacher": 1.2, "The Artist": 1.3, "The Worker": 1.4,
                            "The Box-to-box": 2.1, "The Playmaker": 2.2, "The Anchor": 2.3,
                            "The Solo-dribbler": 3.1, "The 4-4-2-fielder": 3.2, "The Star": 3.3,
                            "The Winger": 4.1, "The Defensive-minded": 4.2, "The Inverted": 4.3,
                            "The Leader": 5.1, "The Low-risk-taker": 5.2, "The Physical": 5.3}, inplace = True)

# Filter by the set position
df_playingS_result_pos = df_playing_style[df_playing_style['Position'] == set_position]

# Compare detected positions to validation data
dict_validation_results_pos = validate.create_validation_dataframes(
    df_playingS_result_pos, "Player_name", "name",
    'Playing-style_primary',
    'Playing-style',
    position=set_position,
    binary_playing_style=False)

# Find the resulting dataframe from the dictionary
df_result = dict_validation_results_pos['df_result']
df_correct = dict_validation_results_pos['df_correct']
df_incorrect = dict_validation_results_pos['df_incorrect']

# Compute and show the confusion matrix with accuracy
df_conf = validate.confusion_matrix(df_result, list_all_playingStyle_indices, 'predicted_class', 'actual_class', show_results=True, table_format='latex')

# Get specific position 
validate.drop_conf_matrix_columns(df_conf, set_position)

# Compute confusion matrix metrics
print("Confusion matrix class metrics classification: \n")
df_class_metrics_pos = validate.confusion_matrix_class_metrics(df_conf, dict_playingStyle_indices[set_position], show_results=True, table_format='latex')



            