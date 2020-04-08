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
	above_average = []
	moving_average_1 = []
	
	while i < games_played:
		try:
			fpts.append(cont['stats']['this-season'][i]['fpts']['2'])
			salary.append(cont['stats']['this-season'][i]['salary']['2'])
		except Exception:
			pass
		i += 1

	for n, i in enumerate(salary):
    		if i == '-':
    				salary[n] = 0

	salary = filter(lambda x: x!=None, salary)
	fpts = list(map(float,fpts))
	salary = list(map(float,salary))
	fpts = list(filter(lambda x: x > 0, fpts))
	salary_length = len(salary)
	salary_total = sum(salary)
	length = len(fpts)
	total = sum(fpts)

	def average(total,length):
		if length < 1:
			return 0
		else:
			return round((total / length),2)

	salary_average = average(salary_total, salary_length)
	salary_average1 = round(salary_average,2)
	game_average = average(total, length)
	player_average = round(game_average, 2)


	def player_value(average, salary):
        	return round(average/(salary/1000),2)

	value = player_value(player_average, salary_average1)

	def player_variance(fpts):
		if len(fpts) < 2:
			return 0
		else:
			return round(statistics.variance(fpts), 2)
	player_variance1 = player_variance(fpts)

	def player_std(fpts):
		if len(fpts) < 2:
			return 0
		else:
			return statistics.stdev(fpts)

	def player_mad(fpts):
		if len(fpts) < 2:
			return 0
		else:
			series = pd.Series(fpts)
			result = series.mad()
			return result

	
	player_mad1 = player_mad(fpts)
	player_std1 = player_std(fpts)
	fpts.reverse()
	
	def moving_average(fpts):
		N=3
		if len(fpts) < 2:
			return [0,0]
		else:
			return (np.convolve(fpts, np.ones((N,))/N, mode='valid'))

	moving_average_1.append(moving_average(fpts))
	reverse_moving_average = moving_average_1[::-1]
	moving_avg = reverse_moving_average[-1]
	final_avg = moving_avg[-1]
	final_moving_average = round(final_avg,2)
	rounded_std = round(player_std1,2)
	rounded_mad = round(player_mad1,2)

	players_list.append([player_one_name, player_team, player_pos, rounded_std, rounded_mad, player_average, value])

	print(player_one_name)
	print('Average: ', game_average)
	print('3 game average: ', final_moving_average)
	print('Standard Deviation: ', rounded_std)
	print('Value: ', value)
	print(' ')
df = pd.DataFrame(players_list, columns = ["Name","Team","Position","Standard Deviation", "Mean Absolute Deviation", "Average", "Value"])
writer = pd.ExcelWriter('MLB_Players.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
df.style.set_properties(**{'text-align':'center'})
pd.set_option('display.max_colwidth',100)
pd.set_option('display.width', 1000)
#print(df)
writer.save()