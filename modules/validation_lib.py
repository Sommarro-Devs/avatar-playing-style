#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:58:19 2022

@author: JakobEP & emildanielsson

Program description:
    
    Library of functions used for validation.
    
"""


#%%
# - Imports
"---------------------------------------------------------------------------"

# Basics
import pandas as pd
import numpy as np

# Nice tables
from tabulate import tabulate

# Project module
import avatarplayingstyle_module.helpers_lib as helper


#%%
# -Get rid of warnings
"---------------------------------------------------------------------------"
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#%%
# - Read in data
"---------------------------------------------------------------------------"

file_directory_data = '../data/processed/'
file_name_data = 'Valideringsunderlag_v2_07_04_22.xlsx'

# Validation data 
df_validation = pd.read_excel(file_directory_data + file_name_data)

# Drop duplicates
df_validation.drop_duplicates(subset=['Player_name'], inplace=True)


#%%
# - Function
"---------------------------------------------------------------------------"

def validate_clustering(df):
    """
    Function which validates clustering results in df for all players in df
    with df_validation.
    
    Intended to be used on results from model_playingStyle.
    
    :param pd.DataFrame df: A pandas dataframe, need to contain column 
                            named 'XXXXXXXX'.
    
    :returns: A dataframe with validation results.
    """
    
    # - Join/merge results with validation data into one df
    "-----------------------------------------------------------------------"

    df_val_result = pd.merge(df, df_validation, how='inner', left_on=['name'], right_on=['Player_name'])

    df_val_result.drop(columns=['Player_name'], inplace=True)
    
    # Create new filtered df 
    df_val_result_processed = df_val_result[['name', 'club', 'Position', 'Clustering_result', 'Playing-style_primary', 'Playing-style_secondary']]
    
    
    return df_val_result_processed


#%%
# - Function
"---------------------------------------------------------------------------"

def validate_positions(df):
    """
    Function which validates computed playing positions for all players in df
    with their "true" playing positions in df_validation.
    
    Intended to be used on results from _________
    
    :param pd.DataFrame df: A pandas dataframe, need to contain column 
                            named 'name' and 'computed position'.                        
    
    :returns: XXXXXXXXXXX
    """
    
    # Read out "true" positions
    df_true_pos = df_validation[['Player_name', 'Position']].copy()
    
    # Filter out players to compare
    df_true_pos = df_true_pos[df_true_pos['Player_name'].isin(df['Player_name'])]
    
    # Compare with computed positions
    df_result = df.compare(df_true_pos, keep_shape=True)
    
    return df_result

#%%
# - Function
"---------------------------------------------------------------------------"
    
    
def rename_pm_positions(df, column_name):
    """
    Function which renames str values in column named 'column_name' to 
    postions FB, CM, CM, OW and ST.
    
    Intended to be used on postions from Playmaker exports.
    
    :param pd.DataFrame df: A pandas dataframe, need to contain column 
                            named 'column_name'.                        
    
    :returns: Nothing, does change inplace on df.
    """
    
    # Map position names correct (only for Playmaker positions)
    df[column_name].replace({'RB': 'FB', 'LB': 'FB', 'DM': 'CM', 
                             'OMC': 'CM', 'RM': 'OW', 'LM': 'OW',
                             'LW': 'OW', 'RW': 'OW'}, inplace=True)

    
#%%
# - Function
"---------------------------------------------------------------------------"
    
def confusion_matrix(df, list_entries, res_predicted, res_actual, show_results = True, table_format = 'grid'):
    """
    Function which creates confusion matrix from results stored in df.
    
    Intended to be for validating position detection algorithm as well as
    computed playing styles against validation set.
    
    :param pd.DataFrame df: A pandas dataframe, need to contain column 
                            named 'res_predicted' and 'res_actual'.      
    :param list list_entries: A list with entries over what the confusion 
                              matrix will include (as index and columns).
    :param str res_computed: Name of computed results to validate.
    :param str res_true: Name of true results to validate.
    :param bool show_results: Bool to determine if resulting dataframe should 
                                be printed or not.
    
    :returns: Computed confusion matrix as a dataframe.
    """
    
    # Initiate confusion matrix
    df_conf = pd.DataFrame(data=0, columns=list_entries, index=list_entries)
    
    # Loop through all results
    for k, result_k in df.iterrows():
        
        # Find computed and true values
        actual_value = result_k[res_actual]
        predicted_value = result_k[res_predicted]
        
        # Write to confusion matrix and count up
        df_conf.at[actual_value, predicted_value] = df_conf.loc[actual_value, predicted_value] + 1
    
    # Do sum on both axis to get total nrs
    df_conf.loc['#predicted'] = df_conf.sum(axis=0)
    df_conf['#actual'] = df_conf.sum(axis=1)
    
    # Find diagonal elements
    diagonal_conf = np.diag(df_conf)
    
    # Intiate list for accuracy
    accuracy_list_predicted = []
    accuracy_list_actual = []
    
    df_conf['Acc_actual'] = 0
    df_conf.loc['Acc_predicted'] = 0
    
    # Loop through diagonal elements
    for k in range(0, len(diagonal_conf)):
        
        # Compute accuracy
        accuracy_k_predicted = round(diagonal_conf[k] / df_conf.loc['#predicted'][k], 2)
        
        accuracy_k_actual = round(diagonal_conf[k] / df_conf['#actual'][k], 2)
        
        # Save and/or add values
        accuracy_list_actual.append(accuracy_k_actual)
        accuracy_list_predicted.append(accuracy_k_predicted)
        
        #df_conf['Acc_actual'][k] = accuracy_k_actual
        #df_conf.loc['Acc_predicted'][k] = accuracy_k_predicted
        
        #df_conf.loc['Accuracy'] = 0
        
        
    # Get total accuracy 
    tot_accuracy = (round(sum(diagonal_conf[ :-1]) / diagonal_conf[-1], 2))
    
    accuracy_list_actual.append(tot_accuracy)
    accuracy_list_predicted.append(tot_accuracy)
    
    # Add accuracy to confusion matrix
    df_conf['Acc_actual'] = accuracy_list_actual
    df_conf.loc['Acc_predicted'] = accuracy_list_predicted
    
    if show_results:
        print(tabulate(df_conf, headers='keys', showindex=True, tablefmt=table_format))
    
    return df_conf
    

#%%
# - Function
"---------------------------------------------------------------------------"
    
def confusion_matrix_class_metrics(df_conf, list_classes, show_results = True, table_format = 'grid'):
    """
    Function which computes statistical measurements for the classes in the 
    confusion matrix.
    
    Intended to be used after a confusion matrix have been created from the 
    function: confusion_matrix().
    
    :param pd.DataFrame df_conf: A pandas dataframe, should be the resulting
                                dataframe from confusion_matrix().      
    :param list list_classes: A list with the confusion matrix classes are.
    :param bool show_results: Bool to determine if resulting dataframe should 
                                be printed or not.
                              
    
    :returns: Dataframe with precision, recall, specificity and F1-score for 
                each class in the confusion matrix.
    """
    
    # Find columns and rows
    list_columns = df_conf.columns.tolist()
    list_rows = df_conf.index.tolist()
    
    # Initiate df
    df_class_metrics = pd.DataFrame()
    
    # Initiate list with total metrics
    list_tot_TP = []
    list_tot_FP = []
    list_tot_FN = []
    list_tot_TN = []
    
    # Loop through all classes
    for i, class_i in enumerate(list_classes):
        
        # - Find TP, TN, FP, FN for class_i
        "------------------------------------------------------------------------"
        
        # True positive
        TP_i = df_conf[class_i][class_i]
        
        # True negative
        df_i = df_conf.copy()
        
        # List classes to keep
        list_classes_i = list_classes.copy()
        list_classes_i.remove(class_i)
        
        # Find columns and rows to drop
        columns_to_drop = list(set(list_columns) - set(list_classes_i))
        rows_to_drop = list(set(list_rows) - set(list_classes_i))
        
        # Drop not used columns
        df_i.drop(columns_to_drop, inplace=True, axis=1)
        df_i.drop(rows_to_drop, inplace=True, axis=0)
        
        # Sum to compute true negative
        TN_i = df_i.values.sum()
        
        # False positive
        FP_i = df_conf[list_classes].loc[class_i].values.sum() - TP_i
        
        # False negative
        FN_i = df_conf.loc[list_classes][class_i].values.sum() - TP_i
        
        
        # - Compute metrics for class_i
        "------------------------------------------------------------------------"
    
        precision_i = round(TP_i / (TP_i + FP_i), 2)
        recall_i = round(TP_i / (TP_i + FN_i), 2)
        F1_score_i = round((2 * TP_i) / (2 * TP_i + FP_i + FN_i), 2)
        specificity_i = round(TN_i / (TN_i + FP_i), 2)
        
    
        # - Save metrics for class_i
        "------------------------------------------------------------------------"
        
        # Nr values
        list_tot_TP.append(TP_i)
        list_tot_FP.append(FP_i)
        list_tot_FN.append(FN_i)
        list_tot_TN.append(TN_i)
        
        
        # Metrics
        df_class_metrics.at[class_i, 'precision'] = precision_i
        df_class_metrics.at[class_i, 'recall'] = recall_i
        df_class_metrics.at[class_i, 'specificity'] = specificity_i
        df_class_metrics.at[class_i, 'F1-score'] = F1_score_i
    
    
    ACCURACY = round(sum(list_tot_TP) / (sum(list_tot_TP) + sum(list_tot_FP)), 2)
    
    if show_results:
        print(f"Accuracy: {ACCURACY}")
        print(tabulate(df_class_metrics, headers='keys', showindex=True, tablefmt=table_format))

    
    return df_class_metrics
    


#%%
# - Function
"---------------------------------------------------------------------------"

def create_validation_dataframes(df_predicted, column_name_validation, 
                                 column_name_predicted, column_name_validation_comparable,
                                 column_name_predicted_comparable, position = None,
                                 binary_playing_style = False):
    """
    Function which creates dataframes that can be used for validation.
    
    Intended to be used early in validation process. The value to the 
    resulting key "df_result" can be used as input to confusion_matrix().
    
    :param string column_name_validation: Column name of the "true/actual" item/player(?)
                                        to be compared. Most often "Player_name".
    :param string column_name_predicted: Column name of the predicted item/player(?)
                                        to be compared. Most often "player" or "name".
    :param string column_name_validation_comparable: Name of the column that contains 
                                                the information of the "true/actual"
                                                classification in df_actual.
                                                Typically "Position" or "Playing-style_primary"
    :param string column_name_predicted_comparable: Name of the column that contains 
                                                the information of the predicted
                                                classification in df_predicted.
                                                Typically "Cluster" or "Playing_style".
    :param string position: String of position used, only needed for binary_classification_style.
    :param bool binary_playing_style: Bool to determine if the created dataframe
                                        comes from binary classification of 
                                        playing styles. 
                                        Do not set to True for position validation.
                              
    
    :returns: Dictionary with three items: 
                {
                "df_result" -  A pandas dataframe, shows the complete resulting 
                                dataframe from the validation.
                "df_correct" - A pandas dataframe, shows the correctly classified
                                items/players.
                "df_incorrect" - A pandas dataframe, shows the incorrectly classified
                                items/players.              
                }
                
    """
    
    #df_validation = df_validation.copy()
    # Change classes for playing styles if binary playing style classification (Offensive/Defensive)
    if binary_playing_style:
        
        df_validation_new = helper.map_binary_playing_style_old(df_validation, position)

    else:
        df_validation_new = df_validation.copy()
        
    # Read out "true/actual" classifications from validation data set
    df_true_pos = df_validation_new[[column_name_validation, column_name_validation_comparable]].copy()
    
    # Rename column
    df_predicted.rename(columns={column_name_predicted: 'name',
                                 column_name_predicted_comparable: 'predicted_class'}, inplace=True)
    
    # Drop duplicates
    df_predicted.drop_duplicates(subset=['name'], inplace=True)
    
    # Initiate counters 
    nr_correct = 0
    nr_incorrect = 0
    nr_tot = 0
    
    # Prepare dataframe with columns to store result
    df_result = pd.DataFrame()
    
    # Loop through all players to validate
    for i, player_i in df_predicted.iterrows():
        
        # Read out player_i computed position
        predicted_class_i = player_i['predicted_class']
        
        # Find player_i in validation data
        mask_player = df_true_pos[column_name_validation] == player_i['name']
        df_true_player = df_true_pos[mask_player]
        
        # Check so player exists
        if len(df_true_player) != 0:
            
            # Increment count
            nr_tot += 1
        
            # Get "true" position for player_i
            actual_class_i = df_true_player[column_name_validation_comparable].values[0]
            
            # Check if computed/predicted class is correct
            if predicted_class_i == actual_class_i:
                
                # Increment count
                nr_correct += 1
                
            # Computed positions is not correct    
            else:
                
                # Increment count
                nr_incorrect += 1
                
            # Save that players name, predicted class and actual class
            # Create temp. df with results for player_i
            df_i = pd.DataFrame([[player_i['name'], predicted_class_i, actual_class_i]], columns=['name', 'predicted_class', 'actual_class'])
                
            # Add to resulting df
            df_result = pd.concat([df_i, df_result], ignore_index=True)
                
    
    # Compute accuracy 
    # accuracy = round(nr_correct/nr_tot, 2)
    # print(f"Accuracy from validation: {accuracy}\n")
    
    # Find which players are assigned correct and incorrect
    mask_correctness = (df_result['predicted_class'] == df_result['actual_class'])
        
    df_correct = df_result[mask_correctness]
    df_incorrect = df_result[~mask_correctness]
    
    
    return {
        "df_result": df_result,
        "df_correct": df_correct,
        "df_incorrect": df_incorrect
        }


def drop_conf_matrix_columns(df_conf, position):
    """
    Function which deletes columns from df_conf depending on given position. 
    
    :param pd.DataFrame df_conf: A pandas dataframe, should be the resulting
                                dataframe from confusion_matrix().     
    :param string position: String of position used

                              
    
    :returns: -
                
    """
    
    # Inititate lists with playingstyles
    list_all_playing_styles = [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3,
                                3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3]
    list_ST_playingstyles = [1.1, 1.2, 1.3, 1.4]
    list_CM_playingstyles = [2.1, 2.2, 2.3]
    list_OW_playingstyles = [3.1, 3.2, 3.3]
    list_FB_playingstyles = [4.1, 4.2, 4.3]
    list_CB_playingstyles = [5.1, 5.2, 5.3]
    
    # initiate set with all playingstyles
    set_all = set(list_all_playing_styles)
    
    # Drop columns in accordance with given  position
    if (position == 'ST'):
        set_position = set(list_ST_playingstyles)
        list_to_drop = list(set_all.difference(set_position))
        df_conf.drop(list_to_drop, axis=1, inplace=True)
        
    if (position == 'CM'):
        set_position = set(list_CM_playingstyles)
        list_to_drop = list(set_all.difference(set_position))
        df_conf.drop(list_to_drop, axis=1, inplace=True)
        
    if (position == 'OW'):
        set_position = set(list_OW_playingstyles)
        list_to_drop = list(set_all.difference(set_position))
        df_conf.drop(list_to_drop, axis=1, inplace=True)
        
    if (position == 'FB'):
        set_position = set(list_FB_playingstyles)
        list_to_drop = list(set_all.difference(set_position))
        df_conf.drop(list_to_drop, axis=1, inplace=True)
        
    if (position == 'CB'):
        set_position = set(list_CB_playingstyles)
        list_to_drop = list(set_all.difference(set_position))
        df_conf.drop(list_to_drop, axis=1, inplace=True)


































