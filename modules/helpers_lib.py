#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: JakobEP & emildanielsson

Program description:
    
    Library of functions that at the moment dont fit in to a more defined library.
"""


#%%
# - Imports
"---------------------------------------------------------------------------"

# Basics
from pathlib import Path
import sys, os
import pandas as pd
import json
import numpy as np

# Standarnization
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# Clustering 
from sklearn.cluster import KMeans

# Project module
import avatarplayingstyle_module.viz_lib as viz


#%%
# - Functions that helps with position detection 
"---------------------------------------------------------------------------"

def map_clusters_to_positions(df_cluster_centers):
    """
    Function which maps clustering results of players to corresponding positions.
    
    :param DataFrame df_cluster: A pandas DataFrame with clustering results for players.
    :param DataFrame df_cluster_centers: A pandas DataFrame with clustering center coordinates.
    
    :returns: Dictionary with mapped positions for all the cluster. 
                
    """
    
    # Find the goalkeeper cluster (lowest x-cordinates)
    GK_cluster_mask = df_cluster_centers.x_center == df_cluster_centers.x_center.min()
    GK_cluster = df_cluster_centers[GK_cluster_mask]['Cluster'].values[0]
    
    # Find the fullbacks cluster (lowest y-cordinates)
    FB_cluster_mask = df_cluster_centers.y_center == df_cluster_centers.y_center.min()
    FB_cluster = df_cluster_centers[FB_cluster_mask]['Cluster'].values[0]
    
    # Copy df_cluster_centers and drop GK and FB 
    df_cluster_centers_filtered = df_cluster_centers.copy()
    df_cluster_centers_filtered.drop(df_cluster_centers_filtered[df_cluster_centers_filtered.Cluster.isin([GK_cluster, FB_cluster])].index, inplace=True)
    
    # Find the center backs cluster (new lowest x-cordinates)
    CB_cluster_mask = df_cluster_centers_filtered.x_center == df_cluster_centers_filtered.x_center.min()
    CB_cluster = df_cluster_centers_filtered[CB_cluster_mask]['Cluster'].values[0]
    
    # drop CB
    df_cluster_centers_filtered.drop(df_cluster_centers_filtered[df_cluster_centers_filtered.Cluster == CB_cluster].index, inplace=True)
    
    # Find the wingers cluster (new lowest y-cordinates)
    OW_cluster_mask = df_cluster_centers_filtered.y_center == df_cluster_centers_filtered.y_center.min()
    OW_cluster = df_cluster_centers_filtered[OW_cluster_mask]['Cluster'].values[0]
    
    # drop OW
    df_cluster_centers_filtered.drop(df_cluster_centers_filtered[df_cluster_centers_filtered.Cluster == OW_cluster].index, inplace=True)

    # Find CM (new lowest x-cordinates)
    CM_cluster_mask = df_cluster_centers_filtered.x_center == df_cluster_centers_filtered.x_center.min()
    CM_cluster = df_cluster_centers_filtered[CM_cluster_mask]['Cluster'].values[0]
    
    # Find ST (new highest x-cordinates)
    ST_cluster_mask = df_cluster_centers_filtered.x_center == df_cluster_centers_filtered.x_center.max()
    ST_cluster = df_cluster_centers_filtered[ST_cluster_mask]['Cluster'].values[0]
    
    
    # create dictionary with positions and map
    dict_cluster_to_position = {
        GK_cluster: 'GK',
        CB_cluster: 'CB',
        FB_cluster: 'FB',
        CM_cluster: 'CM',
        OW_cluster: 'OW',
        ST_cluster: 'ST'
        }
    
    # Return dictionary which mappes the clusters to the positions
    return dict_cluster_to_position


def find_positions(df_all_actions, df_KPI, kpis_for_clustering = [],
                   only_passes_actions = False, scaler = MinMaxScaler(),
                   show_plot = True):
    
    """
    Function which given match events and season-kpis estimates player positions.
    
    :param DataFrame df_all_actions: A pandas DataFrame with actions
        tagged/parsed by PlayMaker.
    :param DataFrame df_KPI: Season KPIs of players.
    :param List<String> kpis_for_clustering: List of the kpis used for 
        clustering dimensions.
    :param only_passes_actions bool: Boolean to decide if avg xpos and ypos 
        should be reduced to only include passes.
    :param scaler sklearn.preprocessing class: Standarnization method.
    :param bool show_plot: Boolean to plot clusters or not.
    
    :returns: DataFrame with resulting position for each player. 
                
    """
    
    # - If we only want to look at the action "passes", overwrite df_all_actions
    "---------------------------------------------------------------------------"
    if only_passes_actions:
        mask_passes = (df_all_actions.action == 'Passes accurate') | (df_all_actions.action == 'Passes (inaccurate)')
        df_all_actions = df_all_actions[mask_passes]
    
    
    # - Group by player -> find avg x and y coord -> reflect to lower pitch half
    "---------------------------------------------------------------------------"
    
    # Find avg position , Group by can be wrong here since multiple events on same time
    df_avg_pos_player = df_all_actions.groupby(by='player', as_index=False)[['xpos', 'ypos']].mean()
    
    # Reflect actions on upper half down to lower
    df_avg_pos_player.loc[df_avg_pos_player['ypos'] > 50, 'ypos'] = 50 - abs(50 - df_avg_pos_player['ypos'])
    
    
    # - Add more dimensions to cluster
    "---------------------------------------------------------------------------"
    
    # Initiate list with what to cluster
    list_clustering = ['xpos', 'ypos'] + kpis_for_clustering
    
    # Choose which KPIs to include
    df_KPI_selected = df_KPI[['name'] + kpis_for_clustering].copy()
    
    # Rename column
    df_KPI_selected.rename(columns={'name': 'player'}, inplace=True)
    
    # Merge KPIs and avg positions
    df_cluster = pd.merge(df_avg_pos_player, df_KPI_selected, how='inner', on='player')
    
    
    # - Standardization of data
    "---------------------------------------------------------------------------"
    
    # Standardization
    df_cluster_std = scaler.fit_transform(df_cluster[list_clustering])

    # Add column names
    df_cluster_std = pd.DataFrame(df_cluster_std, columns=list_clustering)
    
    
    # - Clustering
    "---------------------------------------------------------------------------"
    
    # Create model
    model_cluster = KMeans(n_clusters=6, random_state=3425, algorithm='full')
    
    # Fit model
    model_cluster.fit(df_cluster_std)
    
    # Add clustering result 
    df_cluster['Cluster'] = model_cluster.predict(df_cluster_std)
    
    
    # - Find cluster centers for xpos and ypos 
    "---------------------------------------------------------------------------"
    
    # Initiate dataframe for cluster centers
    df_cluster_centers = pd.DataFrame()
    
    # Loop through the cluster-centers and add to dataframe
    for i, center in enumerate(model_cluster.fit(df_cluster_std).cluster_centers_):
        
        # Find cluster center
        x_center = center[0] # xpos
        y_center = center[1] # ypos
        
        # Add the center coordinates to dataframe
        df_center_i = pd.DataFrame([[i,x_center,y_center]], columns=['Cluster', 'x_center', 'y_center'])
        df_cluster_centers = pd.concat([df_center_i, df_cluster_centers], ignore_index=True)
        
    # Call to function to map cluster to positions
    dict_cluster_to_position = map_clusters_to_positions(df_cluster_centers)
        
     
    # - Create final dataframe with player and computed position
    "---------------------------------------------------------------------------"
    
    # Map positions
    df_cluster['Cluster'].replace({0: dict_cluster_to_position[0],
                                     1: dict_cluster_to_position[1],
                                     2: dict_cluster_to_position[2],
                                     3: dict_cluster_to_position[3],
                                     4: dict_cluster_to_position[4],
                                     5: dict_cluster_to_position[5]},
                                    inplace=True)
    
    # Read out positions from postion dectection results in the cluster dataframe
    df_positions = df_cluster[['player', 'Cluster']].copy()
    
    # - If show_plot is true show the plot
    "---------------------------------------------------------------------------"
    
    # Call plot function for 2-dim cluster plot
    if show_plot:
        viz.plot_clusters_2D(df_cluster, 'xpos', 'ypos', logo_path="../figures_used/Playmaker_Logo.png")
    
    return df_positions


def map_binary_clustering(df_clusters, column_name_cluster, position):
    """
    Function which maps binary clustering (0 or 1) to the classes 
    Offenive or Defensive.
    
    :param DataFrame df_clusters: A pandas DataFrame with results from 
            clustering algoritm.
    :param string column_name_cluster: Name of column with clustering result.
                                        Typically "Clustering_results"
    :param string position: String of position used when clustering.
    
    :returns: DataFrame where cluster is mapped to Offensive or Defensive. 
                
    """
    
    
    # Might need kpi data as input to other position-dependent functiton
    # Or PCA analysis to find characheristic kpis for offenive and defensive
    
    # Call to position funciton to find what should be offensive and defensive
    first_cluster = 'Defensive'     # Determined by function position-dependent
    second_cluster = 'Offensive'    # Determined by function position-dependent
    
    # Map cluster to playing_style
    df_clusters[column_name_cluster].replace({1: first_cluster, 0: second_cluster}, inplace=True)
    
    return df_clusters

# Might not be in use?????
def map_clustering(df_clusters, column_name_cluster, position):
    """
    Function which maps clustering  to the given classes. in validation data.
    
    :param DataFrame df_clusters: A pandas DataFrame with results from 
            clustering algoritm.
    :param string column_name_cluster: Name of column with clustering result.
                                        Typically "Clustering_results"
    :param string position: String of position used when clustering.
    
    :returns: DataFrame where cluster is mapped to Offensive or Defensive. 
                
    """
    
    
    # Might need kpi data as input to other position-dependent functiton
    # Or PCA analysis to find characheristic kpis for offenive and defensive
    
    # Call to position funciton to find what should be offensive and defensive
    first_cluster = 'Defensive'     # Determined by function position-dependent
    second_cluster = 'Offensive'    # Determined by function position-dependent
    
    # Map cluster to playing_style
    df_clusters[column_name_cluster].replace({0: first_cluster, 1: second_cluster}, inplace=True)
    
    return df_clusters


def map_binary_playing_style(df_validation, position):
    """
    Function which maps the validation data of playing styles to binary 
    Offensive/Defensive for each position. 
    
    :param DataFrame df_validation: A pandas DataFrame with validation data.
    :param string position: Position which determines the binary mapping.
    
    :returns: None (overwrites/translates df_validation) 
                
    """
    
    offensive = 'Offensive'
    defensive = 'Defensive'
    
    df_validation_new = df_validation.copy()
    
    # ST mapping
    if position == 'ST':
        
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            # ST
            1.1: defensive,   # The Target
            1.2: offensive,   # The Artist
            1.3: offensive,   # The Poacher
            1.4: defensive,   # The Worker
            
            # CM
            2.1: defensive,   # Box-to-box
            2.2: defensive,   # The Playmkaer
            2.3: defensive,   # The Anchor
           
            
            # OW
            3.1: offensive,   # The Solo-dribbler
            3.2: defensive,   # The 4-4-2-fielder
            3.3: offensive,   # The Star
            
            # FB
            4.1: defensive,   # The Winger
            4.2: defensive,   # The Defensive-minded
            4.3: defensive,   # The Inverted
            
            # CB
            5.1: defensive,   # The Leader
            5.2: defensive,   # The Low-risk-taker
            5.3: defensive,   # The Physical
            }, inplace=True)
        
    # CM mapping
    elif position == 'CM':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            # ST
            1.1: offensive,   # The Target
            1.2: offensive,   # The Artist
            1.3: offensive,   # The Poacher
            1.4: defensive,   # The Worker
            
            # CM
            2.1: defensive,   # Box-to-box
            2.2: offensive,   # The Playmkaer
            2.3: defensive,   # The Anchor
           
            
            # OW
            3.1: offensive,   # The Solo-dribbler
            3.2: defensive,   # The 4-4-2-fielder
            3.3: offensive,   # The Star
            
            # FB
            4.1: offensive,   # The Winger
            4.2: defensive,   # The Defensive-minded
            4.3: defensive,   # The Inverted
            
            # CB
            5.1: defensive,   # The Leader
            5.2: defensive,   # The Low-risk-taker
            5.3: defensive,   # The Physical
            }, inplace=True)
          
    # OW mapping
    elif position == 'OW':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            # ST
            1.1: offensive,   # The Target
            1.2: offensive,   # The Artist
            1.3: offensive,   # The Poacher
            1.4: defensive,   # The Worker
            
            # CM
            2.1: defensive,   # Box-to-box
            2.2: offensive,   # The Playmkaer
            2.3: defensive,   # The Anchor
           
            
            # OW
            3.1: offensive,   # The Solo-dribbler
            3.2: defensive,   # The 4-4-2-fielder
            3.3: offensive,   # The Star
            
            # FB
            4.1: offensive,   # The Winger
            4.2: defensive,   # The Defensive-minded
            4.3: defensive,   # The Inverted
            
            # CB
            5.1: defensive,   # The Leader
            5.2: defensive,   # The Low-risk-taker
            5.3: defensive,   # The Physical
            }, inplace=True)
        
    # FB mapping
    elif position == 'FB':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            # ST
            1.1: offensive,   # The Target
            1.2: offensive,   # The Artist
            1.3: offensive,   # The Poacher
            1.4: offensive,   # The Worker
            
            # CM
            2.1: offensive,   # Box-to-box
            2.2: offensive,   # The Playmkaer
            2.3: defensive,   # The Anchor
           
            
            # OW
            3.1: offensive,   # The Solo-dribbler
            3.2: offensive,   # The 4-4-2-fielder
            3.3: offensive,   # The Star
            
            # FB
            4.1: offensive,   # The Winger
            4.2: defensive,   # The Defensive-minded
            4.3: offensive,   # The Inverted
            
            # CB
            5.1: defensive,   # The Leader
            5.2: defensive,   # The Low-risk-taker
            5.3: defensive,   # The Physical
            }, inplace=True)
        
    # CB mapping
    elif position == 'CB':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            # ST
            1.1: offensive,   # The Target
            1.2: offensive,   # The Artist
            1.3: offensive,   # The Poacher
            1.4: offensive,   # The Worker
            
            # CM
            2.1: offensive,   # Box-to-box
            2.2: offensive,   # The Playmkaer
            2.3: offensive,   # The Anchor
           
            
            # OW
            3.1: offensive,   # The Solo-dribbler
            3.2: offensive,   # The 4-4-2-fielder
            3.3: offensive,   # The Star
            
            # FB
            4.1: offensive,   # The Winger
            4.2: defensive,   # The Defensive-minded
            4.3: offensive,   # The Inverted
            
            # CB
            5.1: offensive,   # The Leader
            5.2: offensive,   # The Low-risk-taker
            5.3: defensive,   # The Physical
            }, inplace=True)
        
    return df_validation_new


def map_binary_playing_style_old(df_validation, position):
    """
    Function which maps the validation data of playing styles to binary 
    Offensive/Defensive for each position from the validernigs data v1. 
    
    :param DataFrame df_validation: A pandas DataFrame with validation data.
    :param string position: Position which determines the binary mapping.
    
    :returns: None (overwrites/translates df_validation) 
                
    """
    
    offensive = 'Offensive'
    defensive = 'Defensive'
    
    df_validation_new = df_validation.copy()
    
    # ST mapping
    if position == 'ST':
        
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            # ST
            1.1: offensive,   # The Powerforward
            1.2: offensive,   # The Artist
            1.3: offensive,   # The Poacher
            1.4: defensive,   # The Worker
            
            }, inplace=True)
        
    # CM mapping
    elif position == 'CM':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
          
            
            # CM
            2.1: offensive,   # Box-to-box
            2.2: defensive,   # The Playmkaer
            2.3: offensive,   # #10
            2.4: defensive,   # The Anchor
           
            
           
            }, inplace=True)
          
    # OW mapping
    elif position == 'OW':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            
            
            # OW
            3.1: offensive,   # The Solo-dribbler
            3.2: defensive,   # The 4-4-2-fielder
            3.3: offensive,   # The Star
            
            
            }, inplace=True)
        
    # FB mapping
    elif position == 'FB':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({
            
            
            # FB
            4.1: offensive,   # The Winger
            4.2: defensive,   # The Defensive-minded
            4.3: offensive,   # The Box-to-box
            
            
            }, inplace=True)
        
    # CB mapping
    elif position == 'CB':
        # Map playing styles
        df_validation_new['Playing-style_primary'].replace({

            # CB
            5.1: offensive,   # The Sweeper
            5.2: defensive,   # The Leader/Anchar
            5.3: defensive,   # The Physical
            }, inplace=True)
        
    return df_validation_new    
    
    
def single_player_style_scout(df_result_PCA, df_result_PCA_excl, var_player,
                        var_club, 
                        dict_playingStyle_mapper, num_position_playingS,
                        use_ranking_scale, logo_path):
    
    if var_player is None:
        pass
    else:
        # - Get the position of the chosen player
        "---------------------------------------------------------------------------"
        # Get values from the chosen player
        df_player = df_result_PCA[df_result_PCA['name'] == var_player]
    
        # Get the position of the player
        actual_player_pos = df_player['Position'].values[0]
                
    
        
        # - Construct params variable 
        "---------------------------------------------------------------------------"
    
        # Initate params variable
        params = []
        active_params = []
    
        # go trhough dictionary to add the params
        for position_i in dict_playingStyle_mapper:
            
            for PC in dict_playingStyle_mapper[position_i]:
                
                # Only add to params if larger than 0
                if  num_position_playingS[position_i] > 0:
                
                    params.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])
                    
                    # If the active player position
                    if position_i == actual_player_pos:
                        active_params.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])
                        
                
                
        
        # - Find dataframe of players with same position
        "---------------------------------------------------------------------------"
    
        # Find dataframe of position in question
        df_PCA_pos = df_result_PCA[df_result_PCA['Position'] == actual_player_pos]
    
    
        
        # - handle the switching of signs for df_PCA_pos dataframe
        "---------------------------------------------------------------------------"
    
        # check if column should switch sign
        for PC in dict_playingStyle_mapper[actual_player_pos]:
            
            # Add the "Other" column (invertedd PC1 for example)
            if PC == "Other":
                df_PCA_pos["Other"] = df_PCA_pos[dict_playingStyle_mapper[actual_player_pos][PC]['PC']]
                
            # Check if to invert 
            if dict_playingStyle_mapper[actual_player_pos][PC]['inverted']:
                print("inverted")
                df_PCA_pos[PC] = df_PCA_pos[PC]*-1
    
    
    
        
        # - Reanme column and set index
        "---------------------------------------------------------------------------"
    
        # Set index for position dataframe
        df_PCA_pos.set_index(['name', 'Position'], inplace=True)
    
        # Rename column
        df_result_PCA_excl.rename({'Test_Position':'Position'}, axis=1, inplace=True)
    
    
        
        # - Check for ranking scaler
        "---------------------------------------------------------------------------"
    
        # get high and lows ([-1,1]) 
        low = (np.ones(len(params)) * -1).tolist()
        high = np.ones(len(params)).tolist()
    
    
        # If we want to use ranking scale instead (quantiles within the league)
        if use_ranking_scale:
            
            # get high and lows ([0, 1]) 
            low = np.zeros(len(params)).tolist()
            high = np.ones(len(params)).tolist()
            
            # - Start with handling df_PCA_pos
            "--------------------------------------------------"
            
            # transform players in the position
            df_PCA_pos[df_PCA_pos.columns.tolist()] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_PCA_pos)
            
            # - Now handle df_result_excl
            "--------------------------------------------------"
            
            # Initiate dataframe to store quantile values in
            df_result_excl = pd.DataFrame()
            
            # transform players in all other positions 
            for position_i in dict_playingStyle_mapper:
                
                # get dataframe of players position_i data
                df_result_excl_i = df_result_PCA_excl[df_result_PCA_excl['Position'] == position_i]
                
                # check if column should switch sign
                for PC in dict_playingStyle_mapper[position_i]:
                    
                    # Add the "Other" column (invertedd PC1 for example)
                    if PC == "Other":                
                            df_result_excl_i["Other"] = df_result_excl_i[dict_playingStyle_mapper[position_i][PC]['PC']]
                            
                    # check if invert
                    if dict_playingStyle_mapper[position_i][PC]['inverted']:
                        df_result_excl_i[PC] = df_result_excl_i[PC]*-1
                
                # Set index
                df_result_excl_i.set_index(['name', 'Position'], inplace=True)
                
                # Transform
                df_result_excl_i[df_result_excl_i.columns.tolist()] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_result_excl_i)
                
                # merge to dataframe
                df_result_excl = pd.concat([df_result_excl, df_result_excl_i])
    
            # overwrtie df_result_PCA_excl
            df_result_PCA_excl = df_result_excl
    
        else:
            
            # Initiate dataframe to store values in
            df_result_excl = pd.DataFrame()
            
            # transform players in all other positions 
            for position_i in dict_playingStyle_mapper:
                
                # get dataframe of players position_i data
                df_result_excl_i = df_result_PCA_excl[df_result_PCA_excl['Position'] == position_i]
                
                
                
                # check if column should switch sign
                for PC in dict_playingStyle_mapper[position_i]:
                    
                    # Add the "Other" column (inverted PC1 for example)
                    if PC == "Other":
                        df_result_excl_i["Other"] = df_result_excl_i[dict_playingStyle_mapper[position_i][PC]['PC']]
                        
                    # check if inverted
                    if dict_playingStyle_mapper[position_i][PC]['inverted']:
                        df_result_excl_i[PC] = df_result_excl_i[PC]*-1
                        
                
                # Set index
                df_result_excl_i.set_index(['name', 'Position'], inplace=True)
                
                # merge to dataframe
                df_result_excl = pd.concat([df_result_excl, df_result_excl_i])
    
            # overwrtie df_result_PCA_excl
            df_result_PCA_excl = df_result_excl
            
            
    
    
        
        # - Create Dataframe that can contruct the spider for the player 
        "---------------------------------------------------------------------------"
    
        # Get values from the chosen player
        df_player_values = df_PCA_pos.loc[var_player]
        df_player_values_excl = df_result_PCA_excl.loc[var_player]
    
        # Merge them and have only one position column
        df_spider = pd.concat([df_player_values, df_player_values_excl])
    
        
        # - Construct player_values
        "---------------------------------------------------------------------------"
    
        # initate player_values
        player_values = []
    
        for position_i in dict_playingStyle_mapper:
            for PC in dict_playingStyle_mapper[position_i]:
                
                # Find value for this PC-component for this position
                value_pos_i = df_spider.loc[position_i][PC]
                
                # append to player_values1_list
                # Only add to params if larger than 0
                if  num_position_playingS[position_i] > 0:
                    player_values.append(value_pos_i)
    
    
        
        # - Skisses of functions for player spiders
        "---------------------------------------------------------------------------"
    
        """
            Note here that "player_values" and "params" must come in the same order.
            Due to the way the code above is written this should however always be the case.
        """
    
        # Plot the spider
        fig = viz.single_player_playmaker_spider(params, player_values, var_player, var_club, actual_player_pos, logo_path, low, high,  active_params = active_params)    
        
        return fig
    
    
    
def compare_playing_style_scout(df_result_PCA, df_result_PCA_excl, var_player1,
                        var_player2, var_club1, var_club2,
                        dict_playingStyle_mapper, num_position_playingS,
                        use_ranking_scale, logo_path):
    
    # - Get the positions of the chosen players
    "---------------------------------------------------------------------------"
    # Player1
    df_player1 = df_result_PCA[df_result_PCA['name'] == var_player1]

    # Player1 pos
    actual_player_pos1 = df_player1['Position'].values[0]

    # Player2
    df_player2 = df_result_PCA[df_result_PCA['name'] == var_player2]

    # Player2 pos
    actual_player_pos2 = df_player2['Position'].values[0]
            

    
    # - Construct params variable 
    "---------------------------------------------------------------------------"

    # Initate params variable
    params = []
    active_params1 = []
    active_params2 = []

    # go trhough dictionary to add the params
    for position_i in dict_playingStyle_mapper:
        
        for PC in dict_playingStyle_mapper[position_i]:
            
            # Only add to params if larger than 0
            if  num_position_playingS[position_i] > 0:
            
                params.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])
                
                # If the active player position
                if position_i == actual_player_pos1:
                    active_params1.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])
                    
                if position_i == actual_player_pos2:
                    active_params2.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])
                    
            
            
    
    # - Find dataframe of players with same position
    "---------------------------------------------------------------------------"

    # Find dataframe of position in question
    df_PCA_pos1 = df_result_PCA[df_result_PCA['Position'] == actual_player_pos1]
    df_PCA_pos2 = df_result_PCA[df_result_PCA['Position'] == actual_player_pos2]


    
    # - handle the switching of signs for df_PCA_pos dataframe
    "---------------------------------------------------------------------------"

    # check if column should switch sign , Player 1
    for PC in dict_playingStyle_mapper[actual_player_pos1]:
        
        # Add the "Other" column (invertedd PC1 for example)
        if PC == "Other":
            df_PCA_pos1["Other"] = df_PCA_pos1[dict_playingStyle_mapper[actual_player_pos1][PC]['PC']]
            
        # Check if to invert 
        if dict_playingStyle_mapper[actual_player_pos1][PC]['inverted']:
            print("inverted")
            df_PCA_pos1[PC] = df_PCA_pos1[PC]*-1
            
    # check if column should switch sign , Player 1
    for PC in dict_playingStyle_mapper[actual_player_pos2]:
        
        # Add the "Other" column (invertedd PC1 for example)
        if PC == "Other":
            df_PCA_pos2["Other"] = df_PCA_pos2[dict_playingStyle_mapper[actual_player_pos2][PC]['PC']]
            
        # Check if to invert 
        if dict_playingStyle_mapper[actual_player_pos2][PC]['inverted']:
            print("inverted")
            df_PCA_pos2[PC] = df_PCA_pos2[PC]*-1



    
    # - Reanme column and set index
    "---------------------------------------------------------------------------"

    # Set index for position dataframe
    df_PCA_pos1.set_index(['name', 'Position'], inplace=True)
    df_PCA_pos2.set_index(['name', 'Position'], inplace=True)

    # Rename column
    df_result_PCA_excl.rename({'Test_Position':'Position'}, axis=1, inplace=True)


    
    # - Check for ranking scaler
    "---------------------------------------------------------------------------"

    # get high and lows ([-1,1]) 
    low = (np.ones(len(params)) * -1).tolist()
    high = np.ones(len(params)).tolist()


    # If we want to use ranking scale instead (quantiles within the league)
    if use_ranking_scale:
        
        # get high and lows ([0, 1]) 
        low = np.zeros(len(params)).tolist()
        high = np.ones(len(params)).tolist()
        
        # - Start with handling df_PCA_pos
        "--------------------------------------------------"
        
        # transform players in the positions
        df_PCA_pos1[df_PCA_pos1.columns.tolist()] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_PCA_pos1)
        df_PCA_pos2[df_PCA_pos2.columns.tolist()] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_PCA_pos2)
        
        # - Now handle df_result_excl
        "--------------------------------------------------"
        
        # Initiate dataframe to store quantile values in
        df_result_excl = pd.DataFrame()
        
        # transform players in all other positions 
        for position_i in dict_playingStyle_mapper:
            
            # get dataframe of players position_i data
            df_result_excl_i = df_result_PCA_excl[df_result_PCA_excl['Position'] == position_i]
            
            # check if column should switch sign
            for PC in dict_playingStyle_mapper[position_i]:
                
                # Add the "Other" column (inverted PC1 for example)
                if PC == "Other":                
                        df_result_excl_i["Other"] = df_result_excl_i[dict_playingStyle_mapper[position_i][PC]['PC']]
                        
                # check if invert
                if dict_playingStyle_mapper[position_i][PC]['inverted']:
                    df_result_excl_i[PC] = df_result_excl_i[PC]*-1
            
            # Set index
            df_result_excl_i.set_index(['name', 'Position'], inplace=True)
            
            # Transform
            df_result_excl_i[df_result_excl_i.columns.tolist()] = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(df_result_excl_i)
            
            # merge to dataframe
            df_result_excl = pd.concat([df_result_excl, df_result_excl_i])

        # overwrtie df_result_PCA_excl
        df_result_PCA_excl = df_result_excl

    else:
        
        # Initiate dataframe to store quantile values in
        df_result_excl = pd.DataFrame()
        
        # transform players in all other positions 
        for position_i in dict_playingStyle_mapper:
            
            # get dataframe of players position_i data
            df_result_excl_i = df_result_PCA_excl[df_result_PCA_excl['Position'] == position_i]
            
            
            
            # check if column should switch sign
            for PC in dict_playingStyle_mapper[position_i]:
                
                # Add the "Other" column (inverted PC1 for example)
                if PC == "Other":
                    df_result_excl_i["Other"] = df_result_excl_i[dict_playingStyle_mapper[position_i][PC]['PC']]
                    
                # check if inverted
                if dict_playingStyle_mapper[position_i][PC]['inverted']:
                    df_result_excl_i[PC] = df_result_excl_i[PC]*-1
                    
            
            # Set index
            df_result_excl_i.set_index(['name', 'Position'], inplace=True)
            
            # merge to dataframe
            df_result_excl = pd.concat([df_result_excl, df_result_excl_i])

        # overwrtie df_result_PCA_excl
        df_result_PCA_excl = df_result_excl
        
        


    
    # - Create Dataframe that can contruct the spider for the player 
    "---------------------------------------------------------------------------"

    # Get values from the chosen players
    df_player_values1 = df_PCA_pos1.loc[var_player1]
    df_player_values_excl1 = df_result_PCA_excl.loc[var_player1]
    df_player_values2 = df_PCA_pos2.loc[var_player2]
    df_player_values_excl2 = df_result_PCA_excl.loc[var_player2]

    # Merge them and have only one position column
    df_spider1 = pd.concat([df_player_values1, df_player_values_excl1])
    df_spider2 = pd.concat([df_player_values2, df_player_values_excl2])


    
    # - Construct player_values
    "---------------------------------------------------------------------------"

    # initate player values
    player_values1 = []
    player_values2 = []

    for position_i in dict_playingStyle_mapper:
        for PC in dict_playingStyle_mapper[position_i]:
            
            # Find value for this PC-component for this position
            value_pos_i1 = df_spider1.loc[position_i][PC]
            value_pos_i2 = df_spider2.loc[position_i][PC]
            
            # append to player_values1_list
            # Only add to params if larger than 0
            if  num_position_playingS[position_i] > 0:
                player_values1.append(value_pos_i1)
                player_values2.append(value_pos_i2)


    
    # - Skisses of functions for player spiders
    "---------------------------------------------------------------------------"

    """
        Note here that "player_values" and "params" must come in the same order.
        Due to the way the code above is written this should however always be the case.
    """

    # Plot the spider
    # viz.single_player_playmaker_spider(params, player_values1, var_player1, var_club1, actual_player_pos1, logo_path, low, high,  active_params = active_params1)    

    return viz.compare_players_playmaker_spider(params,
                                          player_values1, var_player1, var_club1, actual_player_pos1,
                                          player_values2, var_player2, var_club2, actual_player_pos2,
                                          logo_path, low, high, 
                                          active_params1 = active_params1,
                                          active_params2 = active_params2)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    