 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:58:19 2022

@author: JakobEP and Emil Danielsson

Program description:
    
    In config.py, we place in variables that are used across 
    the project.
    
"""

#%%
# - Available positions, playingstyles, index formatting
"---------------------------------------------------------------------------"

positions = ['ST', 'CM', 'OW', 'FB', 'CB']

dict_playingStyle_indices = {
    'ST': [1.1, 1.2, 1.3, 1.4],
    'CM': [2.1, 2.2, 2.3],
    'OW': [3.1, 3.2, 3.3],
    'FB': [4.1, 4.2, 4.3],
    'CB': [5.1, 5.2, 5.3]    
    }

dict_playingStyle_strings = {
    'ST': ["The Target", "The Poacher", "The Artist", "The Worker"],
    'CM': [ "The Box-to-box", "The Playmaker", "The Anchor"],
    'OW': ["The Solo-dribbler", "The 4-4-2-fielder", "The Star"],
    'FB': ["The Winger", "The Defensive-minded", "The Inverted"],
    'CB': ["The Leader", "The Low-risk-taker", "The Physical"]    
    }

# list of all playing style indices
list_all_playingStyle_indices = [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3,
                                 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3]

list_all_playingStyle_strings = ["The Target", "The Poacher", "The Artist", "The Worker",
                       "The Box-to-box", "The Playmaker", "The Anchor",
                       "The Solo-dribbler", "The 4-4-2-fielder", "The Star",
                       "The Winger", "The Defensive-minded", "The Inverted",
                       "The Leader", "The Low-risk-taker", "The Physical"]




#%%
# - Map from what have been seen from PCA for each position
# - Use 'Other' when want opposite of other PC 
"---------------------------------------------------------------------------"

dict_playingStyle_mapper = {
    'ST': {
        'PC1': {
            'playing_style': "The Target",
            'inverted': False,
            },
        'PC2': {
            'playing_style': "The Artist",
            'inverted': False,
            },
        'PC3': {
            'playing_style': "The Worker",
            'inverted': False,
            },
        'PC4': {
            'playing_style': "The Poacher",
            'inverted': False,
            },
        },
    'CM': {
        'PC1': {
            'playing_style': "The Anchor",
            'inverted': True,
            },
        'PC2': {
            'playing_style': "The Playmaker",
            'inverted': False,
            },
        'PC4': {
            'playing_style': "The Box-to-box",
            'inverted': False,
            },
        },
    'OW': {
        'PC1': {
            'playing_style': "The Star",
            'inverted': True,
            },
        'PC2': {
            'playing_style': "The 4-4-2-fielder",
            'inverted': False,
            },
        'PC5': {
            'playing_style': "The Solo-dribbler",
            'inverted': False,
            },
        },
    'FB': {
        'PC1': {
            'playing_style': "The Winger",
            'inverted': False,
            },
        'PC2': {
            'playing_style': "The Inverted",
            'inverted': False,
            },
        'Other': {
            'PC': 'PC1',
            'playing_style': "The Defensive-minded",
            'inverted': True,
            },
        },
    'CB': {
        'PC1': {
            'playing_style': "The Leader",
            'inverted': False,
            },
        'PC3': {
            'playing_style': "The Low-risk-taker",
            'inverted': False,
            },

        'Other': {
            'PC': 'PC2',
            'playing_style': "The Physical",
            'inverted': True,
            }
        }
    }


#%%
# - List of all available KPIs original
"---------------------------------------------------------------------------"

list_KPIs_playmaker = [
    
    # Hard stats
    'goals',
    'goals 90',
    'ass',
    'ass 90',
    'points 90',
    
    # Offensive
    'xg 90',
    'xga 90',
    'sp xg 90',
    'shots 90',
    'xa 90',
    'dribbles 90',
    'dribbles %',
    'xg share',
    'xg impact',
    'tib 90',
    'dze 90',
    'mean xg',
    
    # Defensive
    'tackles 90',
    'tackles %',
    'chall 90',
    'chall %',
    'g mist 90',
    'mist 90',
    'int 90',
    'dribb past 90',
    'wb oh 90',
    
    # Passing
    'passes 90',
    'passes %',
    'xp %',
    'crosses 90',
    'crosses %',
    'kp 90',
    'pib 90',
    'lb 90',
    'lb %',
    'directness',
    'avg pass dist',
    'avg keypass dist',
    'gain 90',
    'tb 90',
    
    # Neither off/def
    'headers 90',
    'headers %',
    'lost b',
    'avg patch area'
    
    ]


#%%
# - List of currently availible KPIs 
"---------------------------------------------------------------------------"

list_KPIs_ALL = [
    
    # Hard stats
    'goals',
    'goals 90',
    'ass',
    'ass 90',
    'points 90', 
       
    # Offensive 
    'xg 90',
    'sp xg 90',
    'xg share',
    'xg share2',
    'xg impact',
    'mean xg',
    'shots 90', 
    'shots share',
    'xa 90',
    'xa share',
    'dribbles 90',
    'dribbles %',
    'dribbles share', 
    'tib 90',
    'tib share',
    'dze 90',
    'dze share', 
    'prog carries 90',
    'prog carries share',
    
    # Defensive
    'tackles 90',
    'tackles %',
    'tackles 90 Padj',
    'tackles share',
    'chall 90',
    'chall %',
    'chall 90 Padj',
    'chall share',
    'g mist 90',
    'mist 90',
    'int 90',
    'int 90 Padj',
    'int share',
    'lost b',
    'lost b Padj',
    'lost b share',
    'dribb past 90',
    'dribb past 90 Padj',
    'dribb past share',
    'wb oh 90',
    'wb oh 90 Padj',
    'wb oh share',
    'fouls 90',
    'fouls 90 Padj',
    'fouls share',
       
    # Passing
    'passes 90',
    'passes share',
    'passes %',
    'xp %',
    'crosses 90',
    'crosses share',
    'crosses %',
    'kp 90',
    'kp share',
    'pib 90',
    'pib share',
    'lb 90',
    'lb share',
    'lb %',
    'directness',
    'directness share',
    'avg pass dist',
    'avg keypass dist',
    'gain 90',
    'gain share', 
    'tb 90', 
    
    # Neither
    'headers 90',
    'headers share',
    'headers %',
    'avg patch area',
    'avg patch area share',
 
    ]


list_KPIs_ST = [
    
    # Hard stats
    #'goals',
    'goals 90',
    #'ass',
    'ass 90',
    #'points 90', 
       
    # Offensive 
    #'xg 90',
    #'sp xg 90',
    #'xg share',
    'xg share2',
    #'xg impact',
    #'mean xg',
    #'shots 90', 
    'shots share',
    #'xa 90',
    'xa share',
    #'dribbles 90',
    'dribbles %',
    'dribbles share', 
    #'tib 90',
    'tib share',
    #'dze 90',
    'dze share', 
    #'prog carries 90',
    'prog carries share',
    
    # Defensive
    #'tackles 90',
    'tackles %',
    #'tackles 90 Padj',
    'tackles share',
    #'chall 90',
    'chall %',
    #'chall 90 Padj',
    'chall share',
    #'g mist 90',
    #'mist 90',
    #'int 90',
    #'int 90 Padj',
    'int share',
    #'lost b',
    #'lost b Padj',
    #'lost b share',
    #'dribb past 90',
    #'dribb past 90 Padj',
    #'dribb past share',
    #'wb oh 90',
    #'wb oh 90 Padj',
    'wb oh share',
    #'fouls 90',
    #'fouls 90 Padj',
    'fouls share',
       
    # Passing
    #'passes 90',
    'passes share',
    'passes %',
    'xp %',
    #'crosses 90',
    #'crosses share',
    #'crosses %',
    #'kp 90',
    'kp share',
    #'pib 90',
    'pib share',
    #'lb 90',
    #'lb share',
    #'lb %',
    #'directness',
    #'directness share',
    #'avg pass dist',
    #'avg keypass dist',
    #'gain 90',
    'gain share', 
    #'tb 90', 
    
    # Neither
    #'headers 90',
    'headers share',
    'headers %',
    #'avg patch area',
    'avg patch area share',
    
    ]

list_KPIs_CM = [
    
    # Hard stats
    #'goals',
    #'goals 90',
    #'ass',
    #'ass 90',
    #'points 90', 
       
    # Offensive 
    #'xg 90',
    #'sp xg 90',
    #'xg share',
    'xg share2',
    #'xg impact',
    #'mean xg',
    #'shots 90', 
    'shots share',
    #'xa 90',
    'xa share',
    #'dribbles 90',
    'dribbles %',
    'dribbles share', 
    #'tib 90',
    'tib share',
    #'dze 90',
    'dze share', 
    #'prog carries 90',
    'prog carries share',
    
    # Defensive
    #'tackles 90',
    'tackles %',
    #'tackles 90 Padj',
    'tackles share',
    #'chall 90',
    'chall %',
    #'chall 90 Padj',
    'chall share',
    #'g mist 90',
    #'mist 90',
    #'int 90',
    #'int 90 Padj',
    'int share',
    #'lost b',
    #'lost b Padj',
    #'lost b share',
    #'dribb past 90',
    #'dribb past 90 Padj',
    #'dribb past share',
    #'wb oh 90',
    #'wb oh 90 Padj',
    'wb oh share',
    #'fouls 90',
    #'fouls 90 Padj',
    'fouls share',
       
    # Passing
    #'passes 90',
    'passes share',
    'passes %',
    'xp %',
    #'crosses 90',
    'crosses share',
    'crosses %',
    #'kp 90',
    'kp share',
    #'pib 90',
    'pib share',
    #'lb 90',
    'lb share',
    'lb %',
    'directness',
    #'directness share',
    'avg pass dist',
    #'avg keypass dist',
    #'gain 90',
    'gain share', 
    #'tb 90', 
    
    # Neither
    #'headers 90',
    #'headers share',
    #'headers %',
    #'avg patch area',
    'avg patch area share',
    
    ]

list_KPIs_OW = [
    
    # Hard stats
    #'goals',
    'goals 90',
    #'ass',
    'ass 90',
    #'points 90', 
       
    # Offensive 
    'xg 90',
    #'sp xg 90',
    #'xg share',
    'xg share2',
    #'xg impact',
    #'mean xg',
    #'shots 90', 
    'shots share',
    #'xa 90',
    'xa share',
    #'dribbles 90',
    'dribbles %',
    'dribbles share', 
    #'tib 90',
    'tib share',
    #'dze 90',
    'dze share', 
    #'prog carries 90',
    'prog carries share',
    
    # Defensive
    #'tackles 90',
    'tackles %',
    #'tackles 90 Padj',
    'tackles share',
    #'chall 90',
    'chall %',
    #'chall 90 Padj',
    'chall share',
    #'g mist 90',
    #'mist 90',
    #'int 90',
    #'int 90 Padj',
    'int share',
    #'lost b',
    #'lost b Padj',
    'lost b share',
    #'dribb past 90',
    #'dribb past 90 Padj',
    'dribb past share',
    #'wb oh 90',
    #'wb oh 90 Padj',
    'wb oh share',
    #'fouls 90',
    #'fouls 90 Padj',
    'fouls share',
       
    # Passing
    #'passes 90',
    'passes share',
    'passes %',
    'xp %',
    #'crosses 90',
    'crosses share',
    'crosses %',
    #'kp 90',
    'kp share',
    #'pib 90',
    'pib share',
    #'lb 90',
    'lb share',
    'lb %',
    'directness',
    #'directness share',
    #'avg pass dist',
    #'avg keypass dist',
    #'gain 90',
    'gain share', 
    #'tb 90', 
    
    # Neither
    #'headers 90',
    #'headers share',
    #'headers %',
    #'avg patch area',
    'avg patch area share',
    
    ]

list_KPIs_FB = [
    
    # Hard stats
    #'goals',
    #'goals 90',
    #'ass',
    #'ass 90',
    #'points 90', 
       
    # Offensive 
    #'xg 90',
    #'sp xg 90',
    #'xg share',
    'xg share2',
    #'xg impact',
    #'mean xg',
    #'shots 90', 
    #'shots share',
    #'xa 90',
    'xa share',
    #'dribbles 90',
    #'dribbles %',
    'dribbles share', 
    #'tib 90',
    #'tib share',
    #'dze 90',
    'dze share', 
    'prog carries 90',
    #'prog carries share',
    
    # Defensive
    #'tackles 90',
    'tackles %',
    'tackles 90 Padj',
    #'tackles share',
    #'chall 90',
    'chall %',
    'chall 90 Padj',
    #'chall share',
    'g mist 90',
    #'mist 90',
    #'int 90',
    'int 90 Padj',
    #'int share',
    #'lost b',
    'lost b Padj',
    #'lost b share',
    #'dribb past 90',
    #'dribb past 90 Padj',
    'dribb past share',
    #'wb oh 90',
    'wb oh 90 Padj',
    #'wb oh share',
    #'fouls 90',
    'fouls 90 Padj',
    #'fouls share',
       
    # Passing
    #'passes 90',
    'passes share',
    #'passes %',
    'xp %',
    #'crosses 90',
    'crosses share',
    #'crosses %',
    #'kp 90',
    'kp share',
    #'pib 90',
    'pib share',
    'lb 90',
    #'lb share',
    #'lb %',
    'directness',
    #'directness share',
    #'avg pass dist',
    #'avg keypass dist',
    'gain 90',
    #'gain share', 
    #'tb 90', 
    
    # Neither
    #'headers 90',
    #'headers share',
    #'headers %',
    #'avg patch area',
    #'avg patch area share',
    
    ]

list_KPIs_CB = [
    
    # Hard stats
    #'goals',
    #'goals 90',
    #'ass',
    #'ass 90',
    #'points 90', 
       
    # Offensive 
    #'xg 90',
    #'sp xg 90',
    #'xg share',
    'xg share2',
    #'xg impact',
    #'mean xg',
    #'shots 90', 
    #'shots share',
    #'xa 90',
    'xa share',
    #'dribbles 90',
    #'dribbles %',
    #'dribbles share', 
    #'tib 90',
    #'tib share',
    #'dze 90',
    #'dze share', 
    'prog carries 90',
    #'prog carries share',
    
    # Defensive
    #'tackles 90',
    'tackles %',
    'tackles 90 Padj',
    #'tackles share',
    #'chall 90',
    'chall %',
    'chall 90 Padj',
    #'chall share',
    'g mist 90',
    #'mist 90',
    #'int 90',
    'int 90 Padj',
    #'int share',
    #'lost b',
    'lost b Padj',
    #'lost b share',
    #'dribb past 90',
    #'dribb past 90 Padj',
    #'dribb past share',
    #'wb oh 90',
    'wb oh 90 Padj',
    #'wb oh share',
    #'fouls 90',
    'fouls 90 Padj',
    #'fouls share',
       
    # Passing
    #'passes 90',
    'passes share',
    #'passes %',
    'xp %',
    #'crosses 90',
    #'crosses share',
    #'crosses %',
    #'kp 90',
    #'kp share',
    #'pib 90',
    #'pib share',
    'lb 90',
    #'lb share',
    'lb %',
    #'directness',
    #'directness share',
    #'avg pass dist',
    #'avg keypass dist',
    'gain 90',
    #'gain share', 
    #'tb 90', 
    
    # Neither
    'headers 90',
    #'headers share',
    'headers %',
    #'avg patch area',
    #'avg patch area share',
    
    ]

#%%
# - List of OLD availible KPIs 
"---------------------------------------------------------------------------"

OLD_list_KPIs_ST = [
    
    # Hard stats
    #'goals',
    'goals 90',
    #'ass',
    #'ass 90',
    #'points 90',
    
    # Offensive
    #'xg 90',
    #'xga 90',
    #'sp xg 90',
    'xg share2',
    #'shots 90',
    'shots share',
    #'xa 90',
    'xa share',
    'dribbles 90',
    #'dribbles %',
    #'xg share',
    #'xg impact',
    'tib 90',
    'dze 90',
    #'mean xg',
    'prog carries 90',
    
    # Defensive
    'tackles 90',
    #'tackles %',
    'chall 90',
    #'chall %',
    #'g mist 90',
    #'mist 90',
    'int 90',
    #'dribb past 90',
    #'wb oh 90',
    'wb oh share',
    'fouls 90',
    
    # Passing
    #'passes 90',
    #'passes %',
    'passes share',
    'xp %',
    'crosses 90',
    #'crosses %',
    'kp 90',
    'kp share',
    'pib 90',
    #'lb 90',
    #'lb %',
    #'directness',
    'avg pass dist',
    #'avg keypass dist',
    #'gain 90',
    #'tb 90',           # Seem to only be zeros
    
    # Neither off/def
    'headers 90',
    #'headers %',
    #'lost b',
    'avg patch area'
    
    # share kpis
    #'xg per shot',     # Can give inf, INCORRECT
    
    ]

OLD_list_KPIs_CM = [
    
    # Hard stats
    #'goals',
    #'goals 90',
    #'ass',
    'ass 90',
    #'points 90',
    
    # Offensive
    #'xg 90',
    #'xga 90',
    #'sp xg 90',
    'xg share2',
    #'shots 90',
    'shots share',
    #'xa 90',
    'xa share',
    'dribbles 90',
    #'dribbles %',
    #'xg share',
    #'xg impact',
    'tib 90',
    'dze 90',
    #'mean xg',
    'prog carries 90',
    
    # Defensive
    'tackles 90',
    #'tackles %',
    'chall 90',
    #'chall %',
    #'g mist 90',
    #'mist 90',
    'int 90',
    #'dribb past 90',
    #'wb oh 90',
    'wb oh share',
    'fouls 90',
    
    # Passing
    #'passes 90',
    #'passes %',
    'passes share',
    'xp %',
    'crosses 90',
    #'crosses %',
    'kp 90',
    'kp share',
    'pib 90',
    #'lb 90',
    #'lb %',
    'directness',
    'avg pass dist',
    #'avg keypass dist',
    'gain 90',
    #'tb 90',           # Seem to only be zeros
    
    # Neither off/def
    #'headers 90',
    #'headers %',
    #'lost b',
    'avg patch area'
    
    # share kpis
    #'xg per shot',     # Can give inf, INCORRECT
    
    ]

OLD_list_KPIs_FB = [
    
    # Hard stats
    #'goals',
    #'goals 90',
    #'ass',
    #'ass 90',
    #'points 90',
    
    # Offensive
    #'xg 90',
    #'xga 90',
    #'sp xg 90',
    #'shots 90',
    'shots share',
    #'xa 90',
    'xa share',
    'dribbles 90',
    #'dribbles %',
    #'xg share',
    #'xg impact',
    'tib 90',
    'dze 90',
    #'mean xg',
    'prog carries 90',
    
    # Defensive
    'tackles 90',
    #'tackles %',
    'chall 90',
    #'chall %',
    #'g mist 90',
    #'mist 90',
    'int 90',
    'dribb past 90',
    #'wb oh 90',
    'wb oh share',
    'fouls 90',
    
    # Passing
    #'passes 90',
    #'passes %',
    'passes share',
    'xp %',
    'crosses 90',
    #'crosses %',
    #'kp 90',
    'kp share',
    'pib 90',
    'lb 90',
    #'lb %',
    #'directness',
    'avg pass dist',
    'avg keypass dist',
    'gain 90',
    #'tb 90',           # Seem to only be zeros
    
    # Neither off/def
    'headers 90',
    #'headers %',
    'lost b',
    'avg patch area'
    
    # share kpis
    #'xg per shot',     # Can give inf, INCORRECT
    #'xg share2',
    
    ]


OLD_list_KPIs = [
    
    # Hard stats
    #'goals',
    #'goals 90',
    #'ass',
    #'ass 90',
    #'points 90',
    
    # Offensive
    #'xg 90',
    #'xga 90',
    #'sp xg 90',
    'shots 90',
    'xa 90',
    'dribbles 90',
    #'dribbles %',
    #'xg share',
    #'xg impact',
    'tib 90',
    'dze 90',
    #'mean xg',
    
    # Defensive
    'tackles 90',
    #'tackles %',
    'chall 90',
    #'chall %',
    'g mist 90',
    'mist 90',
    'int 90',
    'dribb past 90',
    'wb oh 90',
    
    # Passing
    #'passes 90',
    #'passes %',
    'xp %',
    'crosses 90',
    #'crosses %',
    'kp 90',
    'pib 90',
    'lb 90',
    #'lb %',
    'directness',
    'avg pass dist',
    'avg keypass dist',
    'gain 90',
    #'tb 90',           # Seem to only be zeros
    
    # Neither off/def
    'headers 90',
    #'headers %',
    'lost b',
    'avg patch area',
    
    # share kpis
    #'xg per shot',     # Can give inf, INCORRECT
    'passes share',
    'kp share',
    'wb oh share',
    'xa share',
    'shots share',
    'xg share2',
    
    # new kpis
    'fouls 90',
    'prog carries 90'
    ]


#%%
# - Dictionary to map positions with kpi-settings
"---------------------------------------------------------------------------"

dict_kpi_settings = {
    'ST': list_KPIs_ST,
    'CM': list_KPIs_CM,
    'OW': list_KPIs_OW,
    'FB': list_KPIs_FB,
    'CB': list_KPIs_CB
    }