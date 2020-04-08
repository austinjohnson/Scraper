import requests
import json
import numpy as np
import statistics
import pandas as pd
import xlsxwriter
from webscraping_example import players_name


def my_function(x):
    return list(dict.fromkeys(x))


clean_players_id = my_function(players_name)

players_list = []
for x in clean_players_id:
    url = "https://rotogrinders.com/players/"+x+"?format=json"
    r = requests.get(url)
    cont = r.json()
    player_one_name = cont['player']['name']
    player_team = cont['player']['teamName']
    player_pos = cont['player']['position']
    games_played = len(cont['stats']['this-season'])
    i = 0
    fpts = []
    salary = []
    while i < games_played:
        try:
            fpts.append(cont['stats']['this-season'][i]['fpts']['2'])
            salary.append(cont['stats']['this-season'][i]['salary']['2'])
        except Exception:
            pass
        i += 1

    for n, i in enumerate(fpts):
        if i == '-':
            fpts[n] = 0

    salary = list(filter(None, salary))
    #salary = filter(lambda x: x!=None, salary)
    fpts = list(map(float, fpts))
    salary = list(map(float, salary))
    #fpts = filter(lambda x: x > 0, fpts)
    salary = filter(lambda x: x > 0, salary)
    roi_calc = [int(fpts) / int(salary) for fpts, salary in zip(fpts, salary)]

    def ROI_STD(arg):
        try:
            if len(arg) < 1:
                return 0
            else:
                return statistics.stdev(arg)
        except:
            pass

    def ROI_AVG(arg):
        if len(arg) < 1:
            return 0
        else:
            return sum(arg) / len(arg)

    roi_std = ROI_STD(roi_calc)
    roi_avg = ROI_AVG(roi_calc)

    def sharpe_ratio(average, standard_devaition):
        try:
            return (average/standard_devaition)
        except:
            pass

    def player_std(x):
        return statistics.stdev(x)

    std = player_std(fpts)
    sharpe = sharpe_ratio(roi_avg, roi_std)
    players_list.append(
        [player_one_name, player_team, player_pos, sharpe, std])
    print(player_one_name)
    print(sharpe)
    print(' ')
df = pd.DataFrame(players_list, columns=[
                  "Name", "Team", "Position", "Sharpe Ratio", "STD"])
writer = pd.ExcelWriter('MLB_ROI.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
# print(df)
writer.save()
