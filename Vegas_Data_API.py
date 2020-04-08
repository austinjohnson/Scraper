import requests
import json
import numpy as np
import statistics
import pandas as pd
import xlsxwriter

month = '11'
day = '3'

response = requests.get(
    "https://www.fantasylabs.com/api/sportevents/1/"+month+"_"+day+"_2019/vegas/")


def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


games = []

# jprint(response.json())
games.append(response.json()[0]['HomeTeamShort'])
games.append(response.json()[0]['VisitorTeamShort'])
games.append(response.json()[0]['EventDetails']["Properties"]['HomeVegasRuns'])
games.append(response.json()[0]['EventDetails']
             ["Properties"]['VisitorVegasRuns'])
games.append(response.json()[0]['EventDetails']
             ["Properties"]['HomeGameSpreadCurrent'])
games.append(response.json()[0]['EventDetails']
             ["Properties"]['VisitorGameSpreadCurrent'])

print(games)
# df = pd.DataFrame(games, columns=[
#     "Team", "Implied Total", "Spread"])
# writer = pd.ExcelWriter('Vegas_Data.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='Sheet1', index=False)
# df.style.set_properties(**{'text-align': 'center'})
# pd.set_option('display.max_colwidth', 100)
# pd.set_option('display.width', 1000)
# # print(df)
# writer.save()
