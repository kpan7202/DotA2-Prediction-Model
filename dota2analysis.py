# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:32:10 2017

@author: Kurniawan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

ds = pd.read_csv('dota2data.csv')
headers = list(ds.columns.values)
r, c = ds.shape

heroes = pd.read_csv('dota2heroes.csv')

win = len(ds[ds['radiant_win'] == 1])
lose = len(ds[ds['radiant_win'] == 0])

i = 0
heroesData = []
tagTeam = np.zeros((heroes.shape[0], heroes.shape[0]))
for i in range(1, heroes.shape[0] + 2):
    if i != 24:
        hero = {}
        hero['name'] = (heroes['name'][heroes['id'] == i]).to_string(index=0)
        hero['played'] = np.count_nonzero(ds[str(i)])
        hero['win'] = len(ds[np.logical_or(
                np.logical_and(ds[str(i)] == 1,ds['radiant_win'] == 1), 
                np.logical_and(ds[str(i)] == -1,ds['radiant_win'] == 0) 
                )])
        hero['win_rate'] = hero['win'] / hero['played']
        hero['played_rate'] = hero['played'] / r
        hero['kills_per_game'] = np.sum(ds[str(i) + "_kills"]) / hero['played']#[np.logical_or(ds[str(i)] == 1, ds[str(i)] == -1)]
        hero['deaths_per_game'] = np.sum(ds[str(i) + "_deaths"]) / hero['played']
        hero['assists_per_game'] = np.sum(ds[str(i) + "_assists"]) / hero['played']
        hero['xpm_per_game'] = np.sum(ds[str(i) + "_xpm"]) / hero['played']
        hero['gpm_per_game'] = np.sum(ds[str(i) + "_gpm"]) / hero['played']

#        counterHero = ""
#        counterScore = 0
#        partnerHero = ""
#        partnerScore = 0
        for j in range(1, heroes.shape[0] + 2):
            row = i-1 if i < 24 else i-2 
            col = j-1 if j < 24 else j-2 
            if j == i:
                tagTeam[row, col] = None
            if j != 24 and j != i:
                #partner
                radiantRate = ds[np.logical_and(ds[str(i)] == 1, ds[str(j)] == 1)]
                direRate = ds[np.logical_and(ds[str(i)] == -1, ds[str(j)] == -1)]
                if len(radiantRate) + len(direRate) > 0:
                    score = (len(radiantRate[radiantRate['radiant_win'] == 1]) + len(direRate[direRate['radiant_win'] == 0])) / (len(radiantRate) + len(direRate))                                       
                    tagTeam[row, col] = score
#                    if score > partnerScore:
#                        partnerScore = score
#                        partnerHero = (heroes['name'][heroes['id'] == j]).to_string(index=0)
#
#                #counter
#                radiantRate = ds[np.logical_and(ds[str(i)] == 1, ds[str(j)] == -1)]
#                direRate = ds[np.logical_and(ds[str(i)] == -1, ds[str(j)] == 1)]
#                if len(radiantRate) + len(direRate) > 0:
#                    score = (len(radiantRate[radiantRate['radiant_win'] == 0]) + len(direRate[direRate['radiant_win'] == 1])) / (len(radiantRate) + len(direRate))
#                    if score > counterScore:
#                        counterScore = score
#                        counterHero = (heroes['name'][heroes['id'] == j]).to_string(index=0)
#        
#        hero['partner_hero'] = partnerHero
#        hero['partner_score'] = partnerScore
#        hero['counter_hero'] = counterHero
#        hero['counter_score'] = counterScore
        heroesData.append(hero)
        

heroesData = pd.DataFrame(heroesData)
print(heroesData.describe())

# histogram of heroes win rate 
pl.figure()
pl.hist(heroesData['win_rate'],bins = np.linspace(0,1, 10))
pl.title("Frequency of Heroes Win Rate")
pl.xlabel("Win Rate")
pl.ylabel("Frequency")
pl.show()

# histogram of heroes in pair win rate
pair = np.reshape(tagTeam, (heroes.shape[0]**2))
pl.hist(pair[~np.isnan(pair)], bins = np.linspace(0,1, 10))
pl.title("Frequency of Heroes in Pair Win Rate")
pl.xlabel("Win Rate")
pl.ylabel("Frequency")
pl.show()

# plot top 10 most played heroes
mostPlayedHeroes = heroesData.sort_values(by=['played'], ascending = 0)
pl.figure()
pl.barh(range(10),mostPlayedHeroes['played'].iloc[:10], label = "Matches Played", color = "blue")
pl.barh(range(10), mostPlayedHeroes['win'].iloc[:10], label = "Matches Won", color = "red")
pl.title("Top 10 Most Played Heroes")
pl.xlabel("Number of Matches")
pl.ylabel("Heroes")
pl.legend()
pl.yticks(range(10), mostPlayedHeroes['name'].iloc[:10])
pl.show()

# plot kill death ratio
pl.figure()
pl.scatter(heroesData['kills_per_game'],heroesData['deaths_per_game'])
pl.title("Heroes Kill Death Ratio")
pl.xlabel("Average Kills per Match")
pl.ylabel("Average Deaths per Match")
pl.show()

# plot of matches score
pl.figure()
lose = np.where(ds['radiant_win'] == 0)[0]
win = np.where(ds['radiant_win'] == 1)[0]
pl.scatter(ds['radiant_score'][lose],ds['dire_score'][lose], label = "lose")
pl.scatter(ds['radiant_score'][win],ds['dire_score'][win], c='r', label = "win")
pl.legend()
pl.axis([0,300,0,300])
pl.title("Matches Score")
pl.xlabel("Radiant Score")
pl.ylabel("Dire Score")
pl.show()
