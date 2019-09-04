# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:46:47 2017

@author: Kurniawan
"""

import requests
import json
import time
import pandas as pd

API_KEY = '47C31B0F66010E78F1971A9A688D8D79'
GET_MATCH_HISTORY_URL = 'https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V001/'
GET_MATCH_DETAIL_URL = 'https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V001/'
GET_HEROES_URL = 'https://api.steampowered.com/IEconDOTA2_570/GetHeroes/v0001/'


def saveToCSV(data):
    if data is not None and len(data) > 0:
        fileName = time.strftime("%d-%m-%Y %H.%M.%S", time.localtime())
        print('Saving data to csv: ' + fileName)
        newds = pd.DataFrame(data)
        newds.to_csv(fileName + '.csv',header=True, index=False)
        print('Save successful!')

def getHeroes():
    heroes = []
    url = GET_HEROES_URL + '?key=' + API_KEY 
    response = json.loads(requests.get(url).text)
    if response['result']['status'] == 200:
        heroes = response['result']['heroes']
        for hero in heroes:
            hero['name'] = hero['name'].replace('npc_dota_hero_', '')
    return heroes
    

def getMatchDetail(matchId):
     url = GET_MATCH_DETAIL_URL + '?key=' + API_KEY + '&match_id=' + matchId
     data = {}
     try:
         response = json.loads(requests.get(url).text)
         
         data['match_id'] = matchId
         data['radiant_win'] = 1 if response['result']['radiant_win'] else 0
         data['radiant_score'] = response['result']['radiant_score']
         data['dire_score'] = response['result']['dire_score']
         for i in range(115):
             if i != 24:
                 data[str(i)] = 0
                 data[str(i) + '_kills'] = 0
                 data[str(i) + '_deaths'] = 0
                 data[str(i) + '_assists'] = 0
                 data[str(i) + '_xpm'] = 0
                 data[str(i) + '_gpm'] = 0
         
         for player in response['result']['players']:
             if player['leaver_status'] is not 0:
                 print('Leaver found in a match: ' + str(matchId))
                 return None
             data[str(player['hero_id'])] = 1 if player['player_slot'] < 100 else -1 
             data[str(player['hero_id']) + '_kills'] = player['kills']
             data[str(player['hero_id']) + '_deaths'] = player['deaths']
             data[str(player['hero_id']) + '_assists'] = player['assists']
             data[str(player['hero_id']) + '_xpm'] = player['xp_per_min']
             data[str(player['hero_id']) + '_gpm'] = player['gold_per_min']
     except Exception as e:
         print("Error: " + str(matchId))
         return None
#     print(data)
     return data

def getMatchHistory():
    print("Start time: " + time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))
    startMatchId = None
    i = 0
    data = []
    while 1:
        print('Fecthing data')
        url = GET_MATCH_HISTORY_URL + '?key=' + API_KEY + '&skill=3&game_mode=2&min_players=10'
        if startMatchId is not None:
            url += '&start_at_match_id=' + str(startMatchId)
            
        response = json.loads(requests.get(url).text)
        if response['result']['status'] == 1:
            matches = response['result']['matches']
        
            for match in matches:
               record = getMatchDetail(str(match['match_id']))
               if record is not None:
                   data.append(record)
                   i += 1
               time.sleep(1)
            
            if (len(matches) > 0):
                startMatchId =  matches[len(matches)-1]['match_id'] - 1
                
            if response['result']['results_remaining'] <= 0:
                break
        else:
            print('Error has occured. Retrying..')
            
           
        time.sleep(1)
    saveToCSV(data)    
    print("Total records: {}".format(i))    
    print("End time: " + time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))
    return i


if __name__ == '__main__':
    getMatchHistory()
