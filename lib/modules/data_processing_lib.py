#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:58:19 2022

@author: JakobEP & emildanielsson

Program description:
    
    Library of functions used for processing KPI-data and raw data.
    
"""
# TODO: MAYBe delete this file

#%%
# - Imports
"---------------------------------------------------------------------------"

import os
import pandas as pd

import os
import json

# Project module
from modules.config import dict_playingStyle_strings, list_all_playingStyle_strings

# Standarnization
from sklearn.preprocessing import QuantileTransformer


#%%
# - Functions for KPI data proccesing
"---------------------------------------------------------------------------"

def minutes_df_filter(df, min_minutes):
    """
    Function which processes the dataframe on minutes played.
    
    Intended to be used on exports from playmakerai.
    
    :param pd.DataFrame df: A pandas dataframe, need to contain column 
                            named 'minutes'.
    :param int min_minutes: minimum minutes a player has to have played.
    
    :returns: A modified dataframe.
    """ 
    
    # Mask players with too few minutes played
    mask_minutes = df['minutes'] > min_minutes
    
    # Filter out
    df_filtered = df[mask_minutes]
    
    return df_filtered


#%%
# - Functions for raw data processing
"---------------------------------------------------------------------------"

def series_dict_to_df(series_dict):
    """
    Function which converts series of dictionaries to dataframe.
    
    Intended to be used on raw data exports from playmakerai.
    
    :param series series_dict: A pandas series of dictionaries.
    
    :returns: A created dataframe.
    """ 
    
    # Create list and convert to df
    df = pd.DataFrame(series_dict.tolist())
    
    return df


def get_all_events(directory_seasons, season_name, directory_save_data=None):
    """
    Function which gets all events from match events in .json-file located in 
    'directory_seasons' folder for season 'season_name'.
    
    Intended to be used on raw data exports from playmakerai.
    
    :param str directory_seasons: Path/name to directory with all seasons.
    :param str season_name: Name of season to extract events from.
    :param str directory_save_data: Name of directory to save events as a whole
                                    -json-file, optional
    
    :returns: Resulting dataframe with all events.
    """ 
    
    # Initiate resulting df
    df_result = pd.DataFrame()

    # Loop through all files (all matches) in directory
    for filename_name in os.listdir(directory_seasons + '/' + season_name):
        
        # Check to avoid hidden files
        if not filename_name.startswith('.'):
        
            # Open .json.files
            with open(directory_seasons + '/' + season_name + '/' + filename_name) as f:
                data_match = json.load(f)
                
            # Create dataframe of match
            df_match = pd.DataFrame(data_match)
            
            # Create dataframe of events
            df_events = series_dict_to_df(df_match['events'])
            
            # Add found actions to resulting df
            df_result = pd.concat([df_events, df_result], ignore_index=True)
        
    # Check if data should be saved
    if directory_save_data is not None:
        
        df_result.to_json(f"{directory_save_data}/All_events_{season_name}.json")
            
    return df_result


def df_playing_styles_to_quantile(df_playing_styles):
    """
    Function which rescale the playing style scores to quantiles.
    
    Intended to be used on data from function modules/models_lib.py: map_PCA_scores().
    
    :returns: Resulting dataframe quantile scaled playing style for all players in the dataframe.
    """ 

    df_result = df_playing_styles.copy()

    df_result.drop(list_all_playingStyle_strings, axis = 1, inplace = True)

    for position in dict_playingStyle_strings:

        # Find dataframes of playingstyle in position and not in position 
        df_pos_i = df_playing_styles.loc[df_playing_styles['Position'] == position]
        df_excl_i = df_playing_styles.loc[df_playing_styles['Position'] != position]

        # quantile scale for this position 
        df_pos_i[dict_playingStyle_strings[position]] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_pos_i[dict_playingStyle_strings[position]])

        # other quantile scale for out-of-position players for this playing style
        df_excl_i[dict_playingStyle_strings[position]] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_excl_i[dict_playingStyle_strings[position]])

        # initiate list of columns to keep
        final_columns = ['name', 'Position']
        final_columns.extend(dict_playingStyle_strings[position])
        df_pos_i.drop(columns=[col for col in df_pos_i if col not in final_columns], inplace=True)
        df_excl_i.drop(columns=[col for col in df_excl_i if col not in final_columns], inplace=True)

        # Concatenate
        df_result_pos_i = pd.concat([df_pos_i, df_excl_i])

        # merge with result dataframe
        df_result = df_result.merge(df_result_pos_i, on = ['name', 'Position'])
            
     
    return df_result



