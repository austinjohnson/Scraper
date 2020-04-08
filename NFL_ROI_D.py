import requests
import json
import numpy as np
import statistics as st
import pandas as pd
import xlsxwriter
from webscraping_example_D import players_name
import numpy as np


def my_function(x):
    return list(dict.fromkeys(x))


clean_players_id = my_function(players_name)

players_list = []
for x in clean_players_id:
    url = "https://rotogrinders.com/players/"+x+"?site=fanduel&format=json"
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
            salary.append(cont['stats']['this-season'][i]['salary']['2'])

        except Exception:
            pass
        i += 1

    for n, i in enumerate(fpts):
        if i == '-':
            fpts[n] = 0

    salary = list(filter(None, salary))
    fpts = list(map(float, fpts))
    salary = list(map(float, salary))
    salary = filter(lambda x: x > 0, salary)
    # return ROI for every player
    roi_calc = [int(fpts) / int(salary) for fpts, salary in zip(fpts, salary)]

    def fantasy_points_per_minute(lst1, lst2):
        try:
            return [float(lst1) / float(lst2) for lst1, lst2 in zip(lst1, lst2)]
        except:
            pass

    def Player_standard_deviation(arg):
        try:
            if len(arg) < 1:
                return 0
            else:
                return round(st.stdev(arg), 2)
        except:
            pass

    def ROI_STD(arg):
        try:
            return st.stdev(arg)
        except:
            pass

    def ROI_AVG(arg):
        if len(arg) < 1:
            return 0
        else:
            avg = sum(arg) / len(arg)
            return avg

    roi_std = ROI_STD(roi_calc)
    roi_avg = ROI_AVG(roi_calc)

    def sharpe_ratio(average, standard_devaition):
        try:
            return (average/standard_devaition)
        except:
            pass

    def fantasy_points_average(x):
        try:
            return round(sum(x) / len(x), 2)
        except:
            pass

    def no_zeros_array(array):
        try:
            return list(filter(lambda x: x != 0, array))
        except:
            pass

    clean_array = no_zeros_array(fpts)  # removes all zeros from array
    clean_array.reverse()
    del clean_array[-1]
    player_average = fantasy_points_average(clean_array)
    sharpe = sharpe_ratio(roi_avg, roi_std)

    def rounded_sharpe1(arg):
        try:
            return round(arg, 2)
        except:
            pass

    rounded_sharpe = rounded_sharpe1(sharpe)

    def ceiling(array):
        try:
            if len(array) == 0:
                return 0
            elif len(array) < 2:
                return array[0]
            else:
                return max(array)
        except:
            pass

    def floor(array):
        try:
            if len(array) == 0:
                return 0
            elif len(array) < 2:
                return array[0]
            else:
                return min(array)
        except:
            pass

    player_ceil = ceiling(clean_array)
    player_floor = floor(clean_array)
    player_std = Player_standard_deviation(clean_array)

    def calc_deviation(x, y):
        try:
            if x and y == 0:
                pass
            else:
                dev = (x - y) / y
                return round(dev, 2)
        except:
            pass

    def Average(lst):
        try:
            return sum(lst) / len(lst)
        except:
            pass

    def Last_Three(lst):
        try:
            new_lst = lst[-3:]
            three_game_avg = sum(new_lst) / len(new_lst)
            return round(three_game_avg, 2)
        except:
            pass

    three_game_average = Last_Three(clean_array)

    def calc_player_ceil(lst):
        try:
            avg = round(sum(lst) / len(lst), 2)
            sd = round(st.stdev(lst))
            return avg + sd
        except:
            pass

    def calc_player_floor(lst):
        try:
            avg = round(sum(lst) / len(lst), 2)
            sd = round(st.stdev(lst))
            return avg - sd
        except:
            pass

    player_ceiling = calc_player_ceil(clean_array)
    player_floor1 = calc_player_floor(clean_array)

    def subtraction_val(lst, ceil, floor):
        new_lst = []
        try:
            for x in lst:
                if x > ceil:
                    new_val = x - ceil
                    win_val = x - (new_val / 2)
                    rounded_win_val = round(win_val, 2)
                    new_lst.append(rounded_win_val)
                elif x < floor:
                    new_val1 = floor - x
                    win_val1 = x + (new_val1 / 2)
                    rounded_win_val1 = round(win_val1, 2)
                    new_lst.append(rounded_win_val1)
                else:
                    new_lst.append(x)
        except:
            return [0, 0]
        return new_lst

    clean_array_minus_outliers = subtraction_val(
        clean_array, player_ceiling, player_floor1)

    def create_2d_lst(lst):
        try:
            if len(lst) < 1:
                return [0, 0]
            else:
                return [[i, j] for i, j in enumerate(lst)]
        except:
            pass

    two_d_lst = create_2d_lst(clean_array_minus_outliers)

    X = np.matrix(two_d_lst)[:, 0]
    y = np.matrix(two_d_lst)[:, 1]

    def J(X, y, theta):
        theta = np.matrix(theta).T
        m = len(y)
        predictions = X * theta
        sqError = np.power((predictions-y), [2])
        return 1/(2*m) * sum(sqError)

    two_d_lstX = np.matrix(two_d_lst)[:, 0:1]
    X = np.ones((len(two_d_lstX), 2))
    X[:, 1:] = two_d_lstX

    # gradient descent function

    def gradient(X, y, alpha, theta, iters):
        J_history = np.zeros(iters)
        m = len(y)
        theta = np.matrix(theta).T
        for i in range(iters):
            h0 = X * theta
            delta = (1 / m) * (X.T * h0 - X.T * y)
            theta = theta - alpha * delta
            J_history[i] = J(X, y, theta.T)
        return J_history, theta

    # theta initialization
    theta = np.matrix([np.random.random(), np.random.random()])
    alpha = 0.01  # learning rate
    iters = 2000  # iterations

    # this actually trains our model and finds the optimal theta value
    J_history, theta_min = gradient(X, y, alpha, theta, iters)

    # This function will calculate the predicted profit

    def predict(pop):
        return [1, pop] * theta_min

    # Now
    p = len(two_d_lst)

    prediction = round(predict(p).item(), 2)

    def last_game(lst):
        try:
            return lst[-1]
        except:
            pass

    last_game1 = last_game(clean_array)

    fppm = fantasy_points_per_minute(fpts, minutes)

    players_list.append([player_one_name, player_team,
                         player_pos, player_average, three_game_average, prediction, rounded_sharpe, player_std, last_game1])

    print('Player name: ', player_one_name)
    print('Player Sharpe: ', rounded_sharpe)
    print('Projected Score: ', prediction)
    print('Game Log:              ', clean_array)
    print('Game Log W/O outliers: ', clean_array_minus_outliers)
    print('Standard Deviation: ', player_std)
    print('Average: ', player_average)
    print('Three Game average: ', three_game_average)
    print(' ')

df = pd.DataFrame(players_list, columns=[
    "Name", "Team", "Position", "Average", "Three game average", "Prediction", "Sharpe Ratio", "STD", "Last Game"])
writer = pd.ExcelWriter('NFL_Def.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
# print(df)
writer.save()
