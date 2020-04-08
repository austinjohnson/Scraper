import requests
import json
import numpy as np
import statistics
import pandas as pd
import xlsxwriter
from webscraping_example_WR import players_name


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
    this_season = len(cont['stats']['this-season'])
    last_season = len(cont['stats']['last-season'])
    games_played = this_season + last_season
    i = 0
    fpts = []
    salary = []

    while i < games_played:
        try:
            fpts.append(cont['stats']['this-season'][i]['fpts']['2'])
            # fpts.append(cont['stats']['last-season'][i]['fpts']['2'])
            salary.append(cont['stats']['this-season'][i]['salary']['2'])
            # salary.append(cont['stats']['last-season'][i]['salary']['2'])
        except Exception:
            pass
        i += 1

    def remove_zero(arg):
        if len(arg) < 1:
            return 0
        else:
            return arg.pop(0)

    remove_zero(fpts)
    remove_zero(salary)
    print(fpts)
    print(salary)
    for n, i in enumerate(fpts):
        if i == '-':
            fpts[n] = 0

    salary = list(filter(None, salary))
    fpts = list(map(float, fpts))
    salary = list(map(float, salary))
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
    player_std = ROI_STD(fpts)

    def sharpe_ratio(average, standard_devaition):
        try:
            return (average/standard_devaition)
        except:
            pass

    try:
        week1 = fpts[7]
        week2 = fpts[6]
        week3 = fpts[5]
        week4 = fpts[4]
        week5 = fpts[3]
        week6 = fpts[2]
        week7 = fpts[1]
        week8 = fpts[0]
    except:
        'dnp'

    sharpe = sharpe_ratio(roi_avg, roi_std)
    players_list.append([player_one_name, player_team, player_pos, week1, week2,
                         week3, week4, week5, week6, week7, week8, sharpe, player_std])
    print(player_one_name)
    print(sharpe)
    print(player_std)
    print(' ')
df = pd.DataFrame(players_list, columns=[
                  "Name", "Team", "Position", "week1", "week2", "week3", "week4", "week5", "week6", "week7", "week8", "Sharpe Ratio", "STD"])
writer = pd.ExcelWriter('NFL_ROI_WR.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
# print(df)
writer.save()
