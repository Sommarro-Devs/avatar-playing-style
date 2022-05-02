#!/usr/bin/env python3
# -*-     coding: utf-8 -*-
"""
Created on Mon Feb 7 15:18:14 2022

@author: emildanielsson & JakobEP

Program description:
    
    Library of functions used for plotting.

"""

#%%
# - Imports
"---------------------------------------------------------------------------"

# Basics
from pathlib import Path
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# For images 
from PIL import Image

# Statistical fitting of models
import statsmodels.api as sm

# Project module
import modules.plot_styles as pltStyle
from modules.radar_class import PlayMakerRadar
from modules.config import dict_playingStyle_mapper


#%%
# - Functions
"---------------------------------------------------------------------------"

def playmakerScatterPlot(df_KPI, x_kpi, y_kpi, logo_path, save_path,
                         title = None, sub_title=None,  highlighted_players = None,
                         show_names = True, save_fig = False, player_name_size = 12, offset_hl = [0.1, 0.1], offset_all = [0.0, 0.0]):
    """
    Function that returns a scatter plot given 2 kpis.
    
    Intended to be used to compare relationships between kpis and also to compare
    players.
    
    :param pd.DataFrame df_KPI: A pandas dataframe with KPIs.
    :param string x_kpi: kpi to plot on x-axis.
    :param string y_kpi: kpi to plot on y-axis.
    :param string logo_path: full relative path to playmaker logo png.
    :param strirng save_path: full relative path to save location"
    :param string title: title of the plot (deafaults to "kpi_x vs kpi_y").
    :param string sub_title: sub_title of the plot.
    :param list<string> highlighted_players: list of player-names to highlight in plot.
    :param bool show_names: determines if the plot should show names or not.
    :param bool save_fig: determines if the figure should be saved.
    :param int player_name_size: font size of the player names.
    :param float x_offset: x offset for displayed names.
    :param float y_offset: y offset for displayed names.
    
    :returns: Plots a nice catter plot given 2 kpis with Playmaker.AI-Theme
    """
    
    # - Find Playmaker logo
    "---------------------------------------------------------------------------"
    
    logo_file = Path(logo_path)
    if logo_file.is_file():
        # file exists
        logo_img = Image.open(logo_path)
        
    # - Set default title and subtitle if no other given
    "---------------------------------------------------------------------------"
    if title is None:
        title =  f"{x_kpi} vs {y_kpi}"
        
    if sub_title is None:
        sub_title =  df_KPI['season'][0]
    
    # - Model fitting to find regression line
    "---------------------------------------------------------------------------"
    # Define variables
    x_fit1 = df_KPI[x_kpi]
    y_fit1 = df_KPI[y_kpi]
    
    # Add constant term
    X_fit1 = sm.add_constant(x_fit1)
    
    # Do fitting of model by performing regression
    # OLS = ORDINARY LEAST SQUARES
    model1 = sm.OLS(y_fit1, X_fit1)
    result_fit1 = model1.fit()
    
    # Print results
    print("\n ==================== RESULTS FIT 1 =========================== \n")
    print(result_fit1.summary())

    # - Evaluate fitting
    "---------------------------------------------------------------------------"
    
    # In sample prediction
    y_pred1 = result_fit1.predict(X_fit1)
    
    
    # Find players that are included (with old index from df_excel)
    list_players = df_KPI['name']
    
    # - Plot
    "---------------------------------------------------------------------------"
    
    # Read out nr of coeff and find scale
    #scalex = 1.0/(df_KPI[x_kpi].max() - df_KPI[x_kpi].min())
    #scaley = 1.0/(df_KPI[y_kpi].max() - df_KPI[y_kpi].min())
    with plt.style.context(pltStyle.playmaker_dark):
        # Create figure and axis
        fig, ax = plt.subplots()
        # Scatter plot
        ax.scatter(df_KPI[x_kpi], df_KPI[y_kpi], c=pltStyle.red_playmaker, alpha=0.7, zorder=2)
        
        # Add labels to each point
        for index_j, player_j in list_players.items():
            
            if highlighted_players:
            
                if player_j in highlighted_players:
                    plt.text(x = df_KPI[x_kpi][index_j] + offset_hl[0], y = df_KPI[y_kpi][index_j] + offset_hl[1], s = player_j, 
                             fontdict=dict(color='orange', size=player_name_size + 4, weight='bold'), zorder=3)
                    ax.scatter(df_KPI[x_kpi][index_j], df_KPI[y_kpi][index_j], c='orange', alpha=0.95, zorder=3)
                
            
            
                else: # Could have isin highlighted_players
                    if show_names:
                        plt.text(x = df_KPI[x_kpi][index_j] + offset_all[0], y = df_KPI[y_kpi][index_j] + offset_all[1], s = player_j, 
                             fontdict=dict(size=player_name_size), zorder=3)
            
            else:
                if show_names:
                        plt.text(x = df_KPI[x_kpi][index_j] + offset_all[0], y = df_KPI[y_kpi][index_j] + offset_all[1], s = player_j, 
                             fontdict=dict(size=player_name_size), zorder=3)
            
        
        # x and y-labels
        ax.set_xlabel(x_kpi)
        ax.set_ylabel(y_kpi)
        
        # Plot avg 
        ax.hlines(y=df_KPI[y_kpi].mean(), xmin=df_KPI[x_kpi].min(), 
                   xmax=df_KPI[x_kpi].max(), color='#5bc0de', 
                   label=f"avg {x_kpi}", alpha=0.3, zorder=1)
        
        # Plot avg 
        ax.vlines(x=df_KPI[x_kpi].mean(), ymin=df_KPI[y_kpi].min(), 
                   ymax=df_KPI[y_kpi].max(), color='#df691b', 
                   label=f"avg {y_kpi}", alpha=0.3, zorder=1)
        
        # Plot regression line
        ax.plot(x_fit1, y_pred1, color='grey', label='regression line', alpha=0.5, zorder=1)
        
        # Add legend
        leg = ax.legend()
        for text in leg.get_texts():
            plt.setp(text)
    
        
        # Adding title and subtitle
        fig.text(0.15, 1.02, title + "\n", fontdict=dict(size=24, weight='bold'))
        fig.text(0.15, 1.02, sub_title, fontsize=18)
        
        # Adding logo
        if logo_img:
            ax = fig.add_axes([0.03, 0.98, 0.12, 0.12])
            ax.axis("off")
            ax.imshow(logo_img)
        
        # Write out reference
        fig.text(0.05, -0.015, 'By Jakob EP and Emil D, with data from PlayMaker', fontsize=18,)
        
        plt.tight_layout(pad=2)
        
        # Save fig
        if save_fig:
            plt.savefig(save_path,
                        bbox_inches='tight')
               
    
    # Show Plot
    plt.show()
    
    
def plot_clusters_2D(df_cluster, x_column, y_column, logo_path, show_names=False):
    """
    Function which given some dataframe clustering result plots the clusters in 2D.
    
    :param DataFrame df_cluster: A pandas DataFrame with results from clustering.
    :param String x_column: column name in df_cluster for which to use on x-axis.
    :param String y_column: column name in df_cluster for which to use on y-axis.
    :param string logo_path: full relative path to playmaker logo png.
    :param bool show_names: Boolean to show names next to scatter or not.
    
    :returns: Scatter plot of the players in df_cluster and the clusters as 
                ConvexHulls. 
                
    """
    
    #%%
    # - Find Playmaker logo
    "---------------------------------------------------------------------------"
    
    logo_file = Path(logo_path)
    if logo_file.is_file():
        # file exists
        logo_img = Image.open(logo_path)
    
    
    #%% 
    # - Plot
    "---------------------------------------------------------------------------"
    with plt.style.context(pltStyle.playmaker_dark_and_white):
    
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(16, 12))
        
        
        ax.scatter(df_cluster[x_column], df_cluster[y_column], color='blue', edgecolors="black", linewidths=0.2, s=60, alpha=0.4, zorder=2)
        
        if show_names:
            
            # Get players
            list_players = df_cluster['player']
            
            # Add labels to each point
            for index_j, player_j in list_players.items():
                
                # Add text
                plt.text(x=df_cluster['xpos'][index_j], y=df_cluster['ypos'][index_j], s=player_j, 
                     fontdict=dict(color='black', size=12), zorder=3, alpha=0.7)
          
        # Loop through all clusters to plot
        for i in df_cluster['Cluster'].unique():
            
            # Find all players in that clusters points
            points = df_cluster[df_cluster['Cluster'] == i][[x_column, y_column]].values
            
            # Scatter plot
            #ax.scatter(df_cluster[df_cluster['Cluster'] == i][[x_column]], df_cluster[df_cluster['Cluster'] == i][[y_column]], edgecolors="black", linewidths=0.2, s=60, alpha=0.7, zorder=2)
            
            # Add convex hull for the clusters
            # get convex hull (if possible)
            if len(points) > 2:
                hull = ConvexHull(points)
                # get x and y coordinates
                # repeat last point to close the polygon
                x_hull = np.append(points[hull.vertices, 0],
                                   points[hull.vertices, 0][0])
                y_hull = np.append(points[hull.vertices, 1],
                                   points[hull.vertices, 1][0])
                
                # plot shape
                plt.fill(x_hull, y_hull, alpha=0.14, label=f'Cluster: {i}')
                
                cx = np.mean(hull.points[hull.vertices,0])
                cy = np.mean(hull.points[hull.vertices,1])
                
                plt.plot(cx, cy,  'x')
        
        # x and y-labels
        ax.set_xlabel('Touchline in %')
        ax.set_ylabel('Goal line in %')
        
        # Adding title and subtitle
        fig.text(0.19, 1.02, "Position detection clustering results \n", fontdict=dict(size=28, weight='bold'))
        fig.text(0.19, 1.02, "Actions on upper half reflected down to lower", fontsize=22)
        
        # Add legend
        plt.legend(loc='best', prop={"family": "Times New Roman", 'size': 16}, labelcolor='black')
        
        # Adding logo
        if logo_img:
            ax = fig.add_axes([0.03, 0.98, 0.16, 0.16])
            ax.axis("off")
            ax.imshow(logo_img)
        
        # Write out reference
        fig.text(0.05, -0.015, 'By Jakob EP and Emil D, with data from PlayMaker', fontsize=18, color='white')
        
    
    # Show plot
    plt.show()
    
    
def plot_PCA_screeplot(set_position, league, results_PCA, logo_path):
    """
    Function which plots a PCA screeplot from resulting PCA scores.
    
    :param DataFrame df_cluster: A pandas DataFrame with results from clustering.
    :param string set_position: position which is looked at (just for plot title)
    :param string league: league which is looked at (just for plot title)
    :param decomposition._pca.PCA results_PCA: PCA fitted model.
    
    :returns: PCA screeplot.            
    """
    
    #%%
    # - Find Playmaker logo
    "---------------------------------------------------------------------------"
    
    logo_file = Path(logo_path)
    if logo_file.is_file():
        # file exists
        logo_img = Image.open(logo_path)
    
    
    #%%
    # - Look at principal component (PC) retention and analysis result
    "---------------------------------------------------------------------------"
   
    # Get the component variance
    # Proportion of Variance (from PC1 to PCn (n = number of KPIs))
    variance_ratio_PCA = results_PCA.explained_variance_ratio_

    # Cumulative proportion of variance (from PC1 to PCn)   
    variance_cum_prop_PCA = np.cumsum(results_PCA.explained_variance_ratio_)

    # Find number of PCs
    PC_values = np.arange(results_PCA.n_components_) + 1
    
    # Titles
    title = "PCA Scree Plot\n"
    sub_title = f"{league}, Position: {set_position}"
    
    
    #%%
    # - Plot PCA analysis, scree plot
    "---------------------------------------------------------------------------"
    
    with plt.style.context(pltStyle.playmaker_dark):
    
        # Create figure and axis
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Set ticks size
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # Bar plot
        plt.bar(PC_values, variance_ratio_PCA*100, width=0.8, color='lightblue', label='% Variance')
        
        # Create second y-axis
        axes2 = ax2.twinx()
        axes2.plot(PC_values, variance_cum_prop_PCA*100, color='orange', label='Cummulated % Variance')
        axes2.set_ylim(0, 110)
        axes2.set_ylabel('Cummulated % Variance', fontweight='bold', fontsize=20)
        
        # Set ticks size
        plt.yticks(fontsize=18)
        
        # Add grid and zorder
        ax2.grid(ls="dotted", lw=0.3, color="grey", alpha=1, zorder=1)
        
        # x and y labels
        ax2.set_xlabel('Principal Component', fontweight='bold', fontsize=20)
        ax2.set_ylabel('% Variance Explained', fontweight='bold', fontsize=20)
        ax2.set_xlim(0.5, results_PCA.n_components_ + 1)
        
        # Adding title and subtitle
        fig2.text(0.15, 1.02, title, fontweight='bold', fontsize=24)
        fig2.text(0.15, 1.02, sub_title, fontweight='bold', fontsize=20)
        
        # Add legend
        fig2.legend(loc='right', bbox_to_anchor=(0.95, 0.75), bbox_transform=axes2.transAxes, 
                    prop={"family": "Times New Roman", 'size': 16})
        
        # Remove pips
        ax2.tick_params(axis="both", length=0)
        ax2.xaxis.get_major_ticks()[-1].draw = lambda *args:None
        #ax2.set_xticks(ax2.get_xticks()[1:])
        
        # Adding logo
        if logo_img:
            ax2 = fig2.add_axes([0.03, 0.98, 0.14, 0.14])
            ax2.axis("off")
            ax2.imshow(logo_img)
        
        # Write out reference
        fig2.text(0.05, -0.025, 'By Jakob EP and Emil D, with data from PlayMaker', fontsize=14)
    
        # The tight_layout()
        plt.tight_layout(pad=3)
        
    plt.show()
    

def plot_PCA_weights(df_result_weights, position_var, PCs_to_plot, dict_linear_comb_of_PCs=None, inverted=False,
                     logo_path="../figures_used/Playmaker_Logo.png", 
                     save_path="../reports/figures/PCA/", 
                     title=None, sub_title=None, plot_color=None, save_fig=False, show_legend=True):
    """
    Function that returns a plot over PCA weights for different KPIs.
    
    Intended to be used to compare relationships between KPIs and PCs.
    
    :param pd.DataFrame df_result_weights: A pandas dataframe with PCA weights for each KPI.
    :param string position_var: position to look at.
    :param list<string> PCs_to_plot: list of PCs to plot.
    :param string logo_path: full relative path to playmaker logo png.
    :param string save_path: full relative path to save location"
    :param string title: title of the plot (default to "PCs_to_plot").
    :param string sub_title: sub_title of the plot (default to "position_var").
    :param bool save_fig: determines if the figure should be saved.
    
    :returns: Plots a nice scatter plot over PCA weights for different KPIs.
    """
    
    #%%
    # - Find Playmaker logo
    "---------------------------------------------------------------------------"
    
    logo_file = Path(logo_path)
    if logo_file.is_file():
        # file exists
        logo_img = Image.open(logo_path)
    
    
    #%%
    # - Set default title and subttile if no other given
    "---------------------------------------------------------------------------"
    if title is None:
        title =  f"{PCs_to_plot} "
        
    if sub_title is None:
        sub_title =  f"Position: {position_var}"
    
    
    #%%
    # - Read out what to plot
    "---------------------------------------------------------------------------"
    
    x_KPIs = df_result_weights[df_result_weights['Position'] == position_var]['KPI'].reset_index(drop=True)
    
    
    #%%
    # - Plot
    "---------------------------------------------------------------------------"
    
    with plt.style.context(pltStyle.playmaker_dark):
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Initiate list of combined weights (linear comb)
        if dict_linear_comb_of_PCs is not None:
            y_weights_combined_weights = 0
        
        # Loop through and plot all PCs
        for PC_i in PCs_to_plot:
        
            # Read out PC values
            y_weights = df_result_weights[df_result_weights['Position'] == position_var][PC_i].reset_index(drop=True)
            
            if dict_linear_comb_of_PCs is not None:
                y_weights_combined_weights += y_weights * dict_linear_comb_of_PCs[PC_i]
            
            if inverted:
                y_weights = -y_weights
                PC_i = PC_i + " inverted"
            
            # Plot weight values (with color choices)
            if plot_color:
                # Scatterplot
                ax.scatter(x_KPIs, y_weights, s=180,
                           alpha=0.9, zorder=4, label=PC_i, color=plot_color)
                
                # Lineplot weight values
                ax.plot(x_KPIs, y_weights, '--', alpha=0.6, color=plot_color)
            
            # Plot weight values (with default colors)
            else:
                # Scatterplot
                ax.scatter(x_KPIs, y_weights, s=180,
                           alpha=0.9, zorder=4, label=PC_i)
                
                # Lineplot weight values
                ax.plot(x_KPIs, y_weights, '--', alpha=0.6)
                
        
        if dict_linear_comb_of_PCs is not None:
            # Scatter plot weight combined values
            ax.scatter(x_KPIs, y_weights_combined_weights, s=160,
                       alpha=0.9, zorder=4, label='Linear combination')
            
            # Lineplot weight values
            ax.plot(x_KPIs, y_weights_combined_weights, '-', alpha=0.8)
        
        # Rotate x-axis labels
        plt.xticks(rotation=90)
        
        # Add limit lines
        ax.hlines(y=y_weights.abs().mean(), xmin=x_KPIs[0], xmax=x_KPIs[len(x_KPIs)-1], color='#5bc0de', alpha=0.3, zorder=1)
        ax.hlines(y=-y_weights.abs().mean(), xmin=x_KPIs[0], xmax=x_KPIs[len(x_KPIs)-1], color='#5bc0de', alpha=0.3, zorder=1)
        
        # x- and y-labels and limits
        ax.set_ylabel('Weight', fontweight='bold', fontsize=22)
        ax.set_xlabel('KPIs', fontweight='bold', fontsize=22)
        
        # Add legend
        if show_legend:  
            #leg = ax.legend(markerscale=1)
            ax.legend(markerscale=1)
            # for text in leg.get_texts():
            #     plt.setp(text)
        
        # Adding title and subtitle
        fig.text(0.15, 1.02, title + "\n", fontdict=dict(size=26, weight='bold'))
        fig.text(0.15, 1.02, sub_title, fontsize=22)
        
        # Adding logo
        if logo_img:
            ax = fig.add_axes([0.03, 0.98, 0.14, 0.14])
            ax.axis("off")
            ax.imshow(logo_img)
        
        # Write out reference
        fig.text(0.05, -0.015, 'By Jakob EP and Emil D, with data from PlayMaker', fontsize=18)
        
        plt.tight_layout(pad=2)
        
        # Save fig
        if save_fig:
            plt.savefig(save_path,
                        bbox_inches='tight')
               
    # Show Plot
    plt.show()


def single_player_notebook_plot(df_playing_styles, player_name, num_position_playingS, use_ranking_scale, logo_path):
    # get the player values
    df_player = df_playing_styles.loc[df_playing_styles['name'] == player_name]
    #print(df_player)
    actual_player_pos = df_player['Position'].values[0]

    # Initate params variable
    params = []
    active_params = []

    # go through dictionary to add the params
    for position_i in dict_playingStyle_mapper:
        
        for PC in dict_playingStyle_mapper[position_i]:
            
            # Only add to params if larger than 0
            if  num_position_playingS[position_i] > 0:
            
                params.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])
                
                # If the active player position
                if position_i == actual_player_pos:
                    active_params.append(dict_playingStyle_mapper[position_i][PC]['playing_style'])

    # initate player_values
    player_values = []

    for position_i in dict_playingStyle_mapper:
        for PC in dict_playingStyle_mapper[position_i]:
            
            # Find value for this PC-component for this position
            value_pos_i = df_player[dict_playingStyle_mapper[position_i][PC]['playing_style']].values[0]
            
            # append to player_values
            # Only add to params if larger than 0
            if  num_position_playingS[position_i] > 0:
                player_values.append(value_pos_i)

    # Set high and lows for spider
    # get high and lows ([-1,1]) 
    low = (np.ones(len(params)) * -1).tolist()
    high = np.ones(len(params)).tolist()

    # If we want to use ranking scale instead (quantiles within the league)
    if use_ranking_scale:
        
        # get high and lows ([0, 1]) 
        low = np.zeros(len(params)).tolist()
        high = np.ones(len(params)).tolist()

    # Get the spider figure
    single_player_playmaker_spider(params, player_values, player_name, '', actual_player_pos, logo_path, low, high,  active_params = active_params)

    #return figure_Spider


def single_player_playmaker_spider(params,
                                   player_values,
                                   player,
                                   club,
                                   position,
                                   logo_path,
                                   low,
                                   high, 
                                   active_params = [], 
                                   save_fig = False,
                                   save_path = None):
    
    
    # - Initiate radar
    "---------------------------------------------------------------------------"
    
    radar = PlayMakerRadar(params, low, high,
                  # whether to round any of the labels to integers instead of decimal places
                  round_int=[False]*len(params),
                  num_rings=8,  # the number of concentric circles (excluding center circle)
                  # if the ring_width is more than the center_circle_radius then
                  # the center circle radius will be wider than the width of the concentric circles
                  ring_width=1, center_circle_radius=0.1)
    
    # - Find Playmaker logo
    "---------------------------------------------------------------------------"
    
    logo_file = Path(logo_path)
    if logo_file.is_file():
        # file exists
        logo_img = Image.open(logo_path)
        
        
    # - Find indices of the actual parameters
    "---------------------------------------------------------------------------"
    #active_params_set = set(active_params)
    active_indexes =[]
    for i, e in enumerate(params):
        if e in active_params:
            active_indexes.append(i)
    
    
    # - Plot
    "---------------------------------------------------------------------------"
    
    # styling for params
    active_params_style = {'fontsize' : 27, 'color': 'orange'}
    non_active_params_style = {'fontsize' : 22, 'alpha': 0.8,}

    # use playmaker theme to plot
    with plt.style.context(pltStyle.playmaker_dark):
        fig, axs = radar.setup_axis(facecolor = pltStyle.background_playmaker)

        # initate the radar circle and their styling
        radar.draw_circles(ax=axs, ring_width_scaler = 0.02, facecolor='black', edgecolor='black', alpha=0.2)
        
        # draw the radar
        radar_output = radar.draw_radar(player_values, ax=axs,
                                        kwargs_radar={'facecolor': 'white', 'alpha': 0.4},
                                        kwargs_rings={'facecolor': 'white', 'alpha': 0.0})
                                              
        # draw radar params
        radar.draw_param_labels(ax=axs, active_params = active_params, kwargs_active_text = active_params_style, kwargs_non_active_text = non_active_params_style)
        
        # get outputs from the radar 
        radar_poly, rings_outer, vertices = radar_output
        
        # Plot non active scatters
        non_active_vertices_x = np.delete(vertices[:, 0], active_indexes)
        non_active_vertices_y = np.delete(vertices[:, 1], active_indexes)
        axs.scatter(non_active_vertices_x, non_active_vertices_y,
                             c='white', edgecolors='white', alpha = 0.8, marker='o', s=150, zorder=2)
        
        # Plot active scatters
        active_vertices_x = vertices[active_indexes, 0]
        active_vertices_y = vertices[active_indexes, 1]
        axs.scatter(active_vertices_x, active_vertices_y,
                             c='orange', edgecolors='orange', marker='o', s=150, zorder=2)
                                             
        # Adding title and subtitle
        fig.text(0.95, 1.02, player + ',   ' + '\n', fontsize=25,
                                    ha='right', va='center')
        fig.text(0.97, 1.02, position + '\n', fontsize=25,
                                    ha='right', va='center', c='orange')
        fig.text(0.97, 1.00, club, fontsize=22,
                                    ha='right', va='center', color=pltStyle.red_playmaker)
        
        # Adding logo
        if logo_img:
            ax = fig.add_axes([0.03, 0.98, 0.12, 0.12])
            ax.axis("off")
            ax.imshow(logo_img)
        
        # Write out reference
        fig.text(0.05, -0.015, 'By Jakob EP and Emil D, with data from PlayMaker', fontsize=16,)
        
        plt.tight_layout(pad=2)
        
        # Save fig
        if save_fig:
            plt.savefig(save_path,
                        bbox_inches='tight')
    
    plt.show()
    
    return fig


def compare_players_playmaker_spider(params,
                                   player_values1,
                                   player1,
                                   club1,
                                   position1,
                                   player_values2,
                                   player2,
                                   club2,
                                   position2,
                                   logo_path,
                                   low,
                                   high, 
                                   active_params1 = [], 
                                   active_params2 = [], 
                                   save_fig = False,
                                   save_path = None):
    
    
    # - Initiate radar
    "---------------------------------------------------------------------------"
    
    radar = PlayMakerRadar(params, low, high,
                  # whether to round any of the labels to integers instead of decimal places
                  round_int=[False]*len(params),
                  num_rings=8,  # the number of concentric circles (excluding center circle)
                  # if the ring_width is more than the center_circle_radius then
                  # the center circle radius will be wider than the width of the concentric circles
                  ring_width=1, center_circle_radius=0.1)
    
    # - Find Playmaker logo
    "---------------------------------------------------------------------------"
    
    logo_file = Path(logo_path)
    if logo_file.is_file():
        # file exists
        logo_img = Image.open(logo_path)
        
        
    # - Find indices of the actual parameters
    "---------------------------------------------------------------------------"
    #active_params_set = set(active_params)
    active_indexes1 =[]
    active_indexes2 =[]
    non_active_params = []
    for i, e in enumerate(params):
        if e in active_params1:
            active_indexes1.append(i)
        if e in active_params2:
            active_indexes2.append(i)
        if (e not in active_params1 and e not in active_params2):
            non_active_params.append(e)
    
    
    # - Plot
    "---------------------------------------------------------------------------"
    
    # style for scattter
    scatter_color1 = 'orange'
    scatter_color2 = 'lime'
    if position1 == position2:
        scatter_color2 = scatter_color1
    
    # styling for params
    active_params_style1 = {'fontsize' : 27, 'color': scatter_color1}
    active_params_style2 = {'fontsize' : 27, 'color': scatter_color2}
    non_active_params_style = {'fontsize' : 22, 'alpha': 0.8,}
    hide_params_style = {'fontsize' : 0, 'alpha': 0.0,} # use for params to not show

    # use playmaker theme to plot
    with plt.style.context(pltStyle.playmaker_dark):
        fig, axs = radar.setup_axis(facecolor = pltStyle.background_playmaker)

        # initate the radar circle and their styling
        radar.draw_circles(ax=axs, ring_width_scaler = 0.02, facecolor='black', edgecolor='black', alpha=0.2)
        
        # draw the radar
        radar_output = radar.draw_radar_compare(player_values1, player_values2, ax=axs,
                                        kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                        kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
                                              
        # draw radar params
        if position1 == position2: 
            radar.draw_param_labels(ax=axs, active_params = active_params1,
                                    kwargs_active_text = active_params_style1,
                                    kwargs_non_active_text = non_active_params_style)
        
        # If two different positions
        else:  
            # non active in either positions
            radar.draw_param_labels(ax=axs, active_params = non_active_params,
                                    kwargs_active_text = non_active_params_style,
                                    kwargs_non_active_text = hide_params_style)
            
            # player/position1
            radar.draw_param_labels(ax=axs, active_params = active_params1,
                                    kwargs_active_text = active_params_style1,
                                    kwargs_non_active_text = hide_params_style)
            
            # player/position2
            radar.draw_param_labels(ax=axs, active_params = active_params2,
                                    kwargs_active_text = active_params_style2,
                                    kwargs_non_active_text = hide_params_style)

        
        # get outputs from the radar 
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        
        # Plot non active scatters for player 1
        non_active_vertices_x1 = np.delete(vertices1[:, 0], active_indexes1)
        non_active_vertices_y1 = np.delete(vertices1[:, 1], active_indexes1)
        axs.scatter(non_active_vertices_x1, non_active_vertices_y1,
                             c='white', edgecolors='white', alpha = 0.8, marker='o', s=150, zorder=2)
        
        # Plot active scatters for player 1
        active_vertices_x1 = vertices1[active_indexes1, 0]
        active_vertices_y1 = vertices1[active_indexes1, 1]
        axs.scatter(active_vertices_x1, active_vertices_y1,
                             c=scatter_color1, edgecolors=scatter_color1, marker='o', s=150, zorder=3)
        
        # Plot non active scatters for for player 2
        non_active_vertices_x2 = np.delete(vertices2[:, 0], active_indexes2)
        non_active_vertices_y2 = np.delete(vertices2[:, 1], active_indexes2)
        axs.scatter(non_active_vertices_x2, non_active_vertices_y2,
                             c='white', edgecolors='white', alpha = 0.8, marker='o', s=150, zorder=2)
        
        # Plot active scatters for player 2
        active_vertices_x2 = vertices2[active_indexes2, 0]
        active_vertices_y2 = vertices2[active_indexes2, 1]
        axs.scatter(active_vertices_x2, active_vertices_y2,
                             c=scatter_color2, edgecolors=scatter_color2, marker='o', s=150, zorder=3)
                                             
        # Adding title and subtitle player 1
        fig.text(1.35, 0.60, player1 + ',   ' + '\n', fontsize=25,
                                    ha='right', va='center', c = '#00f2c1')
        fig.text(1.37, 0.60, position1 + '\n', fontsize=25,
                                    ha='right', va='center', c=scatter_color1)
        fig.text(1.37, 0.58, club1, fontsize=22,
                                    ha='right', va='center', color=pltStyle.red_playmaker)
        
        # Adding title and subtitle player 2
        fig.text(1.35, 0.40, player2 + ',   ' + '\n', fontsize=25,
                                    ha='right', va='center', c= '#d80499')
        fig.text(1.37, 0.40, position2 + '\n', fontsize=25,
                                    ha='right', va='center', c=scatter_color2)
        fig.text(1.37, 0.38, club2, fontsize=22,
                                    ha='right', va='center', color=pltStyle.red_playmaker)
        
        # Add title top right
        fig.text(1.35, 1.02,  'Playing-style comparison\n', fontsize=28,
                                    ha='right', va='center', c = 'white', fontweight= 'bold')
        
        # Adding logo
        if logo_img:
            ax = fig.add_axes([0.03, 0.98, 0.12, 0.12])
            ax.axis("off")
            ax.imshow(logo_img)
        
        # Write out reference
        fig.text(0.05, -0.015, 'By Jakob EP and Emil D, with data from PlayMaker', fontsize=16,)
        
        plt.tight_layout(pad=2)
        
        # Save fig
        if save_fig:
            plt.savefig(save_path,
                        bbox_inches='tight')
    
    plt.show()
    
    return fig

