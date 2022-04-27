#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:40:35 2022

@author: emildanielsson
"""

#%%
# - Imports
"---------------------------------------------------------------------------"

from mplsoccer import FontManager

#%%
# - Read fonts
"---------------------------------------------------------------------------"
# Read in fonts
URL1 = ('https://github.com/googlefonts/SourceSerifProGFVersion/blob/main/'
        'fonts/SourceSerifPro-Regular.ttf?raw=true')
serif_regular = FontManager(URL1)

URL2 = ('https://github.com/googlefonts/SourceSerifProGFVersion/blob/main/'
        'fonts/SourceSerifPro-ExtraLight.ttf?raw=true')
serif_extra_light = FontManager(URL2)
URL3 = ('https://github.com/googlefonts/SourceSerifProGFVersion/blob/main/fonts/'
        'SourceSerifPro-Bold.ttf?raw=true')
serif_bold = FontManager(URL3)


#%%
# - Define colors
"---------------------------------------------------------------------------"
color1 = "#313332"
background_white = "white"
background_playmaker = '#2b3e50'
red_playmaker = '#c9272d'
white_color = 'white'
black_color = 'black'
grey_color = "grey"
grass_color = '#46812c'
snow_color = '#fafafa'


#%%
# - Define fonts
"---------------------------------------------------------------------------"
font_bold_prop = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 'larger'} # just for testing


#%%
# - Plot themes
"---------------------------------------------------------------------------"
playmaker_dark = {
    
    # Font 
    "font.family": "serif", 
    "axes.titlesize" : 24,
    "axes.labelsize" : 18,
    "text.color": white_color,
    "axes.labelcolor": white_color,
    "axes.labelweight": 'bold',
    
    # Lines/Scatter
    "lines.linestyle" : '-',
    "lines.linewidth" : 2,
    "lines.markersize": 11,
    "lines.color": red_playmaker,
    #"scatter.edgecolors": black_color,
    "scatter.marker": "o",
    "lines.markerfacecolor": red_playmaker,
    "lines.markeredgecolor":red_playmaker,
    
    # Ticks
    "xtick.labelsize" : 20,
    "ytick.labelsize" : 20,
    "xtick.color": white_color,
    "ytick.color": white_color,
    
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.6,
    "grid.color": grey_color,
    "grid.linewidth": 0.24,
    "grid.linestyle":"-",
    
    # Remove all pips
    "xtick.bottom": False,
    "ytick.left": False,
    
    # Remove spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    
    # Axes colors
    "axes.edgecolor": white_color,
    "axes.facecolor": background_playmaker,
    
    # Figure color and settings
    "figure.facecolor": background_playmaker,
    "figure.edgecolor": background_playmaker,
    "figure.figsize": (16, 12),
    "figure.constrained_layout.use": True,
    
    # Legend
    "legend.loc": "best",
    "legend.facecolor": white_color,
    "legend.framealpha": 0.2,
    "legend.fontsize": 15,
    "legend.markerscale": 2,
    "legend.edgecolor": black_color,
    
    # Savefig
    "savefig.dpi": 260,
    #"savefig.edgecolor": background_playmaker,
    #"savefig.facecolor": background_playmaker,
    
}

playmaker_dark_and_white = {
    
    # Font 
    "font.family": "serif", 
    "axes.titlesize" : 24,
    "axes.labelsize" : 20,
    "text.color": white_color,
    "axes.labelcolor": white_color,
    "axes.labelweight": 'bold',
    
    # Lines/Scatter
    "lines.linestyle" : '-',
    "lines.linewidth" : 2,
    "lines.markersize": 11,
    "lines.color": red_playmaker,
    #"scatter.edgecolors": black_color,
    "scatter.marker": "o",
    "lines.markerfacecolor": red_playmaker,
    "lines.markeredgecolor":red_playmaker,
    
    # Ticks
    "xtick.labelsize" : 20,
    "ytick.labelsize" : 20,
    "xtick.color": white_color,
    "ytick.color": white_color,
    
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.6,
    "grid.color": grey_color,
    "grid.linewidth": 0.24,
    "grid.linestyle":"-",
    
    # Remove all pips
    "xtick.bottom": False,
    "ytick.left": False,
    
    # Remove spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    
    # Axes colors
    "axes.edgecolor": white_color,
    "axes.facecolor": snow_color,
    
    # Figure color and settings
    "figure.facecolor": background_playmaker,
    "figure.edgecolor": background_playmaker,
    "figure.figsize": (16, 12),
    "figure.constrained_layout.use": False,
    "figure.autolayout": True,
    
    # Legend
    "legend.loc": "best",
    "legend.facecolor": white_color,
    "legend.framealpha": 0.2,
    "legend.fontsize": 18,
    "legend.markerscale": 3,
    "legend.edgecolor": black_color,
    
    # Savefig
    "savefig.dpi": 260,
    #"savefig.edgecolor": background_playmaker,
    #"savefig.facecolor": background_playmaker,
    
}
