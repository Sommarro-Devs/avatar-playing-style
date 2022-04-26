#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:31:44 2022

@author: emildanielsson

Program description:
    
    Library of models used in this project.
    
    df_KPI refers to KPI-Dataframe generated from ../create_model_kpi_df.py
    for all model-functions in this script!
    
"""

#%%
# - Imports
"---------------------------------------------------------------------------"

# Basics
import pandas as pd

# Project module
from avatarplayingstyle_module.viz_lib import plot_PCA_screeplot
from avatarplayingstyle_module.config import positions, dict_kpi_settings, dict_playingStyle_mapper

# Statistical fitting of models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#%%
# -Get rid of warnings
"---------------------------------------------------------------------------"
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#%%
# - Read validation data
"---------------------------------------------------------------------------"

file_directory_data = '../data/processed/'
file_name_data = 'Valideringsunderlag_v2_07_04_22.xlsx'

# Validation data 
df_validation = pd.read_excel(file_directory_data + file_name_data)

# Drop duplicates
df_validation.drop_duplicates(subset=['Player_name'], inplace=True)

# Get players which are in validation data
list_players_validation = df_validation['Player_name'].unique()


#%%
# - Model Version 1: Base case counting offensive/defensive actions
"---------------------------------------------------------------------------"

"""
Model-function for base case.

:param pd.DataFrame df_KPI_off_def: A pandas dataframe, generated from file: 
    ../create_KPIs_off_def.py.
:param pd.DataFrame df_KPI: A pandas dataframe, see top of lib-script.
:param dictionary df_KPI: A dictionary which sets the quantile setting for each
                            position.

:returns: A dataframe with model results.
"""
def model_off_def(df_KPI_off_def, df_KPI, position_quantile_mapper):

    #%
    # - Filter and normalize df_KPI_off_def data
    "---------------------------------------------------------------------------"

    # Get only players from validation data
    df_KPI_off_def = df_KPI_off_def[df_KPI_off_def['name'].isin(list_players_validation)]

    # Add columns with player and team normalizations
    df_KPI_off_def['off actions ratio'] = df_KPI_off_def['off actions 90'] / (df_KPI_off_def['off actions 90'] + df_KPI_off_def['def actions 90 Padj'])
    df_KPI_off_def['def actions ratio'] = df_KPI_off_def['def actions 90 Padj'] / (df_KPI_off_def['off actions 90'] + df_KPI_off_def['def actions 90 Padj'])
    
    # NOT IN USE
    # df_KPI_off_def['off actions team ratio'] = df_KPI_off_def['off actions'] / (df_KPI_off_def['off actions team'])
    # df_KPI_off_def['def actions team ratio'] = df_KPI_off_def['def actions'] / (df_KPI_off_def['def actions team'])


    #%
    # - Add position to df_KPI_off_def
    "---------------------------------------------------------------------------"

    for i, player_i in df_KPI.iterrows():
        
        # mask the player in df_KPI_OFF_def
        mask_player = df_KPI_off_def['name'] == player_i['name']

        # Add position to df_KPI_off_def
        df_KPI_off_def.loc[mask_player, 'position'] = player_i['Position_comp'] # could change to 'Position_set'    


    #%
    # - Loop through positions and add off/def classifier to players based on median value
    "---------------------------------------------------------------------------"

    for position in position_quantile_mapper:
        
        # mask for players in the position
        mask_position = df_KPI_off_def['position'] == position
        
        # find dataframe
        df_KPI_off_def_position = df_KPI_off_def[mask_position]
        
        # Find median value
        off_actions_median = df_KPI_off_def_position['off actions ratio'].quantile(q=position_quantile_mapper[position]) 
        
        for i, player_i in df_KPI_off_def_position.iterrows():
            
            # mask the player in df_KPI_off_def
            mask_player = df_KPI_off_def['name'] == player_i['name']
            
            # set player to offensive
            if (player_i['off actions ratio'] >= off_actions_median):
                
                # Set to offensive
                df_KPI_off_def.loc[mask_player, 'playing style'] = 'Offensive'
                
            # Set player to defensive    
            else:
                
                # Set to defensive
                df_KPI_off_def.loc[mask_player, 'playing style'] = 'Defensive'
                
    return df_KPI_off_def


#%%
# - Model Version 2: PCA 
#       
# This model is divided into 2 functions which together form Model v2.
# In between use of the two model-functions, the user should map the PCs
# to playing-styles and use that mapper as input to "map_PCA_scores".
# The mapper created in this project (From PL training data) is used as default
# and is found in config.py as "dict_playingStyle_mapper".
# Use script model2_scheme_example.py to find mapper.
#           
#   *  "create_PCA_dataframes": Creates PCA-score dataframes based on train
#                               data. Returns a dictionary with PC-weights,
#                               PCA-scores for train data correct position,
#                               train data out of position (excl), test data
#                               correct position and test data out of position.  
#
#   *  "map_PCA_scores": Maps PCA-score to platingstyles. Defaults to mapping
#                       accordning to variable "config.dict_playingStyle_mapper"
#                       but should work with other mapping inputs.                                      
"---------------------------------------------------------------------------"

"""
Funtion to create PCA-scores dataframes.

:param pd.DataFrame df_KPI_train: A pandas dataframe, train data for model.
:param pd.DataFrame df_KPI_test: A pandas dataframe, test data for model.
:param int nr_of_PCs: Number of PCs to get scores from.
:param bool screeplot: Show screeplots or not.

:returns: A dictionary with model results.
"""
def create_PCA_scores(df_KPI_train, df_KPI_test, nr_of_PCs = 6, screeplot = False):
    
    # Initiate result dataframes
    df_result_weights = pd.DataFrame()
    df_result_PCA_train = pd.DataFrame()
    df_result_PCA_excl_train = pd.DataFrame()

    df_result_PCA_test = pd.DataFrame()
    df_result_PCA_excl_test = pd.DataFrame()

    # Loop through all positions
    for position_i in positions:
        
        # Get list of kpis for pos_i
        list_KPIs_i = dict_kpi_settings[position_i]

        #%
        # - Filtering and Standardization of KPI train data 
        "---------------------------------------------------------------------------"
        
        # Filter KPI by chosen position
        df_KPI_pos_train = df_KPI_train[df_KPI_train['Position_comp'] == position_i].copy()
        
        # KPIs for all other positions
        df_KPI_excl_train = df_KPI_train[df_KPI_train['Position_comp'] != position_i].copy()
        
        # Mean=0 and Variance=1
        df_KPI_pos_std_train = StandardScaler().fit_transform(df_KPI_pos_train[list_KPIs_i])
        df_KPI_excl_std_train = StandardScaler().fit_transform(df_KPI_excl_train[list_KPIs_i])
        
        # Add column names
        df_KPI_pos_std_train = pd.DataFrame(df_KPI_pos_std_train,
                                            columns = df_KPI_pos_train[list_KPIs_i].columns)
        df_KPI_excl_std_train = pd.DataFrame(df_KPI_excl_std_train,
                                             columns = df_KPI_excl_train[list_KPIs_i].columns)
        
        #%
        # - Filtering and Standardization of KPI test data 
        "---------------------------------------------------------------------------"
        
        # Filter KPI by chosen position
        df_KPI_pos_test = df_KPI_test[df_KPI_test['Position_comp'] == position_i].copy()
        
        # KPIs for all other positions
        df_KPI_excl_test = df_KPI_test[df_KPI_test['Position_comp'] != position_i].copy()
        
        # Mean=0 and Variance=1      
        df_KPI_pos_std_test = StandardScaler().fit_transform(df_KPI_pos_test[list_KPIs_i])
        df_KPI_excl_std_test = StandardScaler().fit_transform(df_KPI_excl_test[list_KPIs_i])
        
        # Add column names
        df_KPI_pos_std_test = pd.DataFrame(df_KPI_pos_std_test,
                                           columns = df_KPI_pos_test[list_KPIs_i].columns)
        df_KPI_excl_std_test = pd.DataFrame(df_KPI_excl_std_test,
                                            columns = df_KPI_excl_test[list_KPIs_i].columns)
             
        
        #%
        # - Perform PCA
        "---------------------------------------------------------------------------"
        
        # Define PCA model and fit it to data
        model_PCA = PCA()
        results_PCA = model_PCA.fit(df_KPI_pos_std_train)
        
        # Component loadings or weights (correlation coefficient between original variables and the component) 
        # Component loadings/weights represents the elements of the eigenvector
        # the squared loadings/weights within the PCs always sums to 1 by definition
        weights = results_PCA.components_
        num_pc = results_PCA.n_features_
        pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
        df_weights = pd.DataFrame.from_dict(dict(zip(pc_list, weights)))
        
        # Change variable names to the KPIs and set index
        df_weights['KPI'] = df_KPI_pos_std_train.columns.values
        df_weights['Position'] = position_i
        # df_weights.set_index(['KPI', 'Position'], inplace=True)
        
        
        #%
        # - Plot PCA analysis, scree plot
        "---------------------------------------------------------------------------"
        
        # Plot screeplot for train data position_i
        if screeplot: 
            plot_PCA_screeplot(position_i, 'PL 2021/2022', results_PCA, logo_path="../figures_used/Playmaker_Logo.png")
        
          
        #%
        # - Look at principal component (PC) retention and analysis result
        "---------------------------------------------------------------------------"
        
        # Get PC scores for actual position
        scores_PCA_train = model_PCA.transform(df_KPI_pos_std_train)
        scores_PCA_test = model_PCA.transform(df_KPI_pos_std_test)
        
        # Get PC scores for the other positions
        scores_PCA_excl_train = model_PCA.transform(df_KPI_excl_std_train)
        scores_PCA_excl_test = model_PCA.transform(df_KPI_excl_std_test)
        
        
        #%
        # - Rescale PC values
        "---------------------------------------------------------------------------"
        
        # df_result_PCA_scaled = df_result_PCA.copy()
        # df_reuslt_PCA_excl_scaled = df_result_PCA_excl.copy()
         
        # Loop through columns to rescale
        for i in range(0, nr_of_PCs):
            
            # Find scalers train data
            scaler_train = 1.0/(scores_PCA_train[:,i].max() - scores_PCA_train[:,i].min())
            scaler_excl_train = 1.0/(scores_PCA_excl_train[:,i].max() - scores_PCA_excl_train[:,i].min())
            
            # scale column pc_i train data
            scores_PCA_train[:,i] = scores_PCA_train[:,i]*scaler_train
            scores_PCA_excl_train[:,i] = scores_PCA_excl_train[:,i]*scaler_excl_train
            
            # Allsvenskan
            scaler_test = 1.0/(scores_PCA_test[:,i].max() - scores_PCA_test[:,i].min())
            scaler_excl_test = 1.0/(scores_PCA_excl_test[:,i].max() - scores_PCA_excl_test[:,i].min())
            
            # scale column pc_i
            scores_PCA_test[:,i] = scores_PCA_test[:,i]*scaler_test
            scores_PCA_excl_test[:,i] = scores_PCA_excl_test[:,i]*scaler_excl_test
        
        
        #%
        # - Set up result collection PCA train data
        "---------------------------------------------------------------------------"
        
        # Initiate resulting dataframe
        df_PCA_train = pd.DataFrame(df_KPI_pos_train['name'].copy())
        df_PCA_excl_train = pd.DataFrame(df_KPI_excl_train['name'].copy())
        
        # Reset and drop index
        df_PCA_train.reset_index(drop=True, inplace=True)
        df_PCA_excl_train.reset_index(drop=True, inplace=True)
        
        # Merge to get KPI/PCA values
        df_PCA_train = pd.concat([df_PCA_train,
                                  pd.DataFrame(scores_PCA_train[:, 0:nr_of_PCs])], axis=1)
        df_PCA_excl_train = pd.concat([df_PCA_excl_train,
                                       pd.DataFrame(scores_PCA_excl_train[:, 0:nr_of_PCs])], axis=1)
        
        # Change column names to PCs
        df_PCA_train.columns.values[1: ] = pc_list[0:nr_of_PCs]
        df_PCA_excl_train.columns.values[1: ] = pc_list[0:nr_of_PCs]
        
        # Add actual position
        df_PCA_train['Position'] = position_i
        
        # Add "test/trying" position
        df_PCA_excl_train['Position_excl'] = position_i
        
        # Set index - WHY DID WE INCLUDE THIS? MAYBE SOME FUNCTION IN OTHER SCRIPT?
        # df_PCA_train.set_index(['name', 'Position'], inplace=True)
        # df_PCA_excl_train.set_index(['name', 'Position_excl'], inplace=True)
        
        
        #%
        # - Save results train data position_i
        "---------------------------------------------------------------------------"
        
        df_result_weights = pd.concat([df_result_weights, df_weights])
        df_result_PCA_train = pd.concat([df_result_PCA_train, df_PCA_train])
        df_result_PCA_excl_train = pd.concat([df_result_PCA_excl_train, df_PCA_excl_train])
        
        
        #%
        # - Set up result collection PCA test data
        "---------------------------------------------------------------------------"
        
        # Initiate resulting dataframe
        df_PCA_test = pd.DataFrame(df_KPI_pos_test['name'].copy())
        df_PCA_excl_test = pd.DataFrame(df_KPI_excl_test['name'].copy())
        
        # Reset and drop index
        df_PCA_test.reset_index(drop=True, inplace=True)
        df_PCA_excl_test.reset_index(drop=True, inplace=True)
        
        # Merge to get KPI/PCA values
        df_PCA_test = pd.concat([df_PCA_test, pd.DataFrame(scores_PCA_test[:, 0:nr_of_PCs])], axis=1)
        df_PCA_excl_test = pd.concat([df_PCA_excl_test, pd.DataFrame(scores_PCA_excl_test[:, 0:nr_of_PCs])], axis=1)
        
        # Change column names to PCs
        df_PCA_test.columns.values[1: ] = pc_list[0:nr_of_PCs]
        df_PCA_excl_test.columns.values[1: ] = pc_list[0:nr_of_PCs]
        
        # Add actual position
        df_PCA_test['Position'] = position_i
        
        # Add "test/trying" position
        df_PCA_excl_test['Position_excl'] = position_i
        
        # Set index - WHY DID WE INCLUDE THIS? MAYBE SOME FUNCTION IN OTHER SCRIPT?
        # df_PCA_test.set_index(['name', 'Position'], inplace=True)
        # df_PCA_excl_test.set_index(['name', 'Position_excl'], inplace=True)
        
        
        #%
        # - Save results Allsvenskan
        "---------------------------------------------------------------------------"
        
        # df_result_weights = pd.concat([df_result_weights, df_weights])
        df_result_PCA_test = pd.concat([df_result_PCA_test, df_PCA_test])
        df_result_PCA_excl_test = pd.concat([df_result_PCA_excl_test, df_PCA_excl_test])
        

    #%
    # - return resulting dataframes
    "---------------------------------------------------------------------------"
    
    return {
        'result_weights': df_result_weights,
        'result_train': df_result_PCA_train,
        'result_excl_train': df_result_PCA_excl_train,
        'result_test': df_result_PCA_test,
        'result_excl_test': df_result_PCA_excl_test
        }
    

"""
Function to map PCA-scores generated from "create_PCA_scores" to playing styles.

:param pd.DataFrame df_PCA_scores: A pandas dataframe, PCA-scores with actual
                                 positions to map.
:param pd.DataFrame df_PCA_scores_excl: A pandas dataframe, PCA-scores out of
                                 positions to map.
:param dictionary dict_mapper: A dictionary with mapping for PCA -> playing styles.

:returns: A dataframe with mapped playing styles / model v2 results.
"""
def map_PCA_scores(df_PCA_scores, df_PCA_scores_excl, dict_mapper = dict_playingStyle_mapper):
    
    #%
    # - Get playing styles from mapper
    "---------------------------------------------------------------------------"

    list_playingS = []

    for position_i in dict_mapper:
        
        for PC in dict_mapper[position_i]:
            
            # add the playingStyle to the list
            list_playingS.append(dict_mapper[position_i][PC]['playing_style'])
                
            
    #%
    # - Convert PCA to the actual playing styles for each player
    "---------------------------------------------------------------------------"

    # initiate resulting columns for the dataframe
    columns_result = ['name', 'Position']
    columns_result.extend(list_playingS)

    # Initiate dataframe to store values in
    df_result = pd.DataFrame(columns = columns_result)
    
    # main loop to map playingstyles
    for i, player in df_PCA_scores.iterrows():
        
        # Get values from the chosen player
        df_player = df_PCA_scores[df_PCA_scores['name'] == player['name']]
        
        # get excl values from the player 
        df_player_excl = df_PCA_scores_excl[df_PCA_scores_excl['name'] == player['name']]
        
        # Get the position of the player
        actual_player_pos = df_player['Position'].values[0]
        
        # Initiate player list result
        player_result_list = [player['name'], actual_player_pos]
        
        
        # - Loop through positions to find dataframe of the player 
        "---------------------------------------------------------------------------"
        for position_i in dict_mapper:
            
            # check if the position is the correct for the player
            if position_i == actual_player_pos:
                
                # check if column should switch sign
                for PC in dict_mapper[position_i]:
                    
                    # Add the "Other" column (invertedd PC1 for example)
                    if PC == "Other":
                        df_player["Other"] = df_player[dict_mapper[position_i][PC]['PC']]
                        
                    # Check if to invert 
                    if dict_mapper[position_i][PC]['inverted']:
                        # print("inverted")
                        df_player[PC] = df_player[PC]*-1
                        
                    # Add PC value to the list
                    player_result_list.append(df_player[PC].values[0])
                
            else:
                
                # get excl values from the player 
                df_player_excl_pos = df_player_excl[df_player_excl['Position_excl'] == position_i]
                
                # check if column should switch sign
                for PC in dict_mapper[position_i]:
                    
                    # Add the "Other" column (invertedd PC1 for example)
                    if PC == "Other":
                        df_player_excl_pos["Other"] = df_player_excl_pos[dict_mapper[position_i][PC]['PC']]
                        
                    # Check if to invert 
                    if dict_mapper[position_i][PC]['inverted']:
                        # print("inverted")
                        df_player_excl_pos[PC] = df_player_excl_pos[PC]*-1
                        
                    # Add PC value to the list
                    player_result_list.append(df_player_excl_pos[PC].values[0])
        
        
        # make dataframe of the player value series
        df_player_result = pd.DataFrame([player_result_list], columns=columns_result)
        
        # append to the dataframe
        df_result = pd.concat([df_result, df_player_result])
        
    #%
    # - get the binary playingStyle
    "---------------------------------------------------------------------------"
                    
    for i, player in df_result.iterrows():
        
        # get position of the player
        player_position = player['Position']
        
        # initate playingStyle and value
        playingS = ""
        playingS_value = -1
        
        # loop through playing styles within the position
        for PC in dict_mapper[player_position]:
            
            # playingStyle
            playing_style_i = dict_mapper[player_position][PC]['playing_style']
            
            if player[playing_style_i] > playingS_value:
                playingS = playing_style_i
                playingS_value = player[playing_style_i]
        
        df_result.loc[df_result['name'] == player['name'], ['Playing-style']] = playingS
        
        # - Get best fitting playing style in the other position groups 
        "---------------------------------------------------------------------------"
        
        # initate playingStyle and value
        playingS = ""
        playingS_value = -1
        for position_i in dict_mapper:
            
            if position_i != player_position:
            
                # loop through playing styles within the position
                for PC in dict_mapper[position_i]:
                    
                    # playingStyle
                    playing_style_i = dict_mapper[position_i][PC]['playing_style']
                    
                    if player[playing_style_i] > playingS_value:
                        playingS = playing_style_i
                        playingS_value = player[playing_style_i]
        
        df_result.loc[df_result['name'] == player['name'], ['Playing-style_excl']] = playingS
        
    # reorder the columns
    columns_new_order = ['name', 'Position', 'Playing-style', 'Playing-style_excl']
    columns_new_order.extend(list_playingS)
    df_result = df_result.reindex(columns = columns_new_order)

    return df_result

#%%
# - Model Version 3: Supervised SGD 
#                                           
"---------------------------------------------------------------------------"

"""
Funtion to create PCA-scores dataframes.

:param pd.DataFrame df_KPI_train: A pandas dataframe, train data for model.
:param pd.DataFrame df_KPI_test: A pandas dataframe, test data for model.
:param int nr_of_PCs: Number of PCs to get scores from.
:param bool screeplot: Show screeplots or not.

:returns: A dictionary with model results.
"""
# def model_sgd(df_KPI_train, df_KPI_test, nr_of_PCs = 6, screeplot = True):
