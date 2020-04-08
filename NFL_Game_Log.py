import requests
import json
import numpy as np
import statistics as st
import pandas as pd
import xlsxwriter
from webscraping_example_NFL import players_name
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from scipy.signal import find_peaks, argrelextrema


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

    pass_yards = []
    rush_yards = []
    receiving_yards = []
    interceptions = []
    fumbles = []
    pass_td = []
    rush_td = []
    receiving_td = []
    receptions = []
    targets = []
    rzatt = []
    rztar = []
    touches = []

    while i < games_played:
        try:
            fpts.append(cont['stats']['this-season'][i]['fpts']['2'])
            salary.append(cont['stats']['this-season'][i]['salary']['2'])
            pass_yards.append(cont['stats']['this-season'][i]['payds'])
            rush_yards.append(cont['stats']['this-season'][i]['ruyds'])
            receiving_yards.append(cont['stats']['this-season'][i]['reyds'])
            interceptions.append(cont['stats']['this-season'][i]['int'])
            fumbles.append(cont['stats']['this-season'][i]['fuml'])
            pass_td.append(cont['stats']['this-season'][i]['patd'])
            rush_td.append(cont['stats']['this-season'][i]['rutd'])
            receiving_td.append(cont['stats']['this-season'][i]['retd'])
            receptions.append(cont['stats']['this-season'][i]['rec'])
            targets.append(cont['stats']['this-season'][i]['tar'])
            rzatt.append(cont['stats']['this-season'][i]['rzatt'])
            rztar.append(cont['stats']['this-season'][i]['rztar'])
            touches.append(cont['stats']['this-season'][i]['tchs'])

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
    two_d_lst_pass_yards = create_2d_lst(pass_yards)  # pass yards
    two_d_lst_rush_yards = create_2d_lst(rush_yards)  # rush yards
    two_d_lst_rec_yards = create_2d_lst(receiving_yards)  # rec yards
    two_d_lst_interceptions = create_2d_lst(interceptions)  # interceptions
    two_d_lst_fumbles = create_2d_lst(fumbles)  # fumbles
    two_d_lst_pass_tds = create_2d_lst(pass_td)  # pass tds
    two_d_lst_rush_tds = create_2d_lst(rush_td)  # rush tds
    two_d_lst_rec_tds = create_2d_lst(receiving_td)  # rec tds
    two_d_lst_receptions = create_2d_lst(receptions)  # rec
    two_d_lst_targets = create_2d_lst(targets)  # targets
    two_d_lst_rzatt = create_2d_lst(rzatt)  # redzone attempts
    two_d_lst_rztar = create_2d_lst(rztar)  # redzone targets
    two_d_lst_touches = create_2d_lst(touches)  # touches

    def Predictor(data):

        X = np.matrix(data)[:, 0]
        y = np.matrix(data)[:, 1]

        def J(X, y, theta):
            theta = np.matrix(theta).T
            m = len(y)
            predictions = X * theta
            sqError = np.power((predictions-y), [2])
            return 1/(2*m) * sum(sqError)

        dataX = np.matrix(data)[:, 0:1]
        X = np.ones((len(dataX), 2))
        X[:, 1:] = dataX

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
        alpha = 0.10  # learning rate
        iters = 10000  # iterations

        # this actually trains our model and finds the optimal theta value
        J_history, theta_min = gradient(X, y, alpha, theta, iters)

        # This function will calculate the predicted profit

        def predict(pop):
            return [1, pop] * theta_min

        # Now
        p = len(data)

        prediction = [predict(p).item(), predict(
            p+1).item(), predict(p+2).item()]
        prediction = sum(prediction) / len(prediction)
        if(prediction < 0):
            return 0
        else:
            return round(prediction, 2)

    projection = Predictor(two_d_lst)
    proj_pass_yds = Predictor(two_d_lst_pass_yards)
    proj_rush_yds = Predictor(two_d_lst_rush_yards)
    proj_rec_yds = Predictor(two_d_lst_rec_yards)
    proj_interceptions = Predictor(two_d_lst_interceptions)
    proj_fumbles = Predictor(two_d_lst_fumbles)
    proj_pass_tds = Predictor(two_d_lst_pass_tds)
    proj_rush_tds = Predictor(two_d_lst_rush_tds)
    proj_rec_tds = Predictor(two_d_lst_rec_tds)
    proj_rec = Predictor(two_d_lst_receptions)
    proj_tar = Predictor(two_d_lst_targets)
    proj_rzatt = Predictor(two_d_lst_rzatt)
    proj_rztar = Predictor(two_d_lst_rztar)
    proj_touches = Predictor(two_d_lst_touches)

    def last_game(lst):
        try:
            return lst[-1]
        except:
            pass

    try:
        rounded_clean_array = [round(x) for x in clean_array]
        formatted_game_log = np.array(rounded_clean_array)

        y = formatted_game_log
        y = np.array(y)
        x = (range(0, (len(formatted_game_log))))
        x = np.array(x)
        colors = (0, 0, 0)
        area = np.pi*3

        dips = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1

        peaks, _ = find_peaks(y)

        peaks = list(peaks)
    except:
        pass

    def filter_local_max_under_avg(game_log, peaks, average_points):
        res = list()
        for idx in peaks:
            if game_log[idx] > average_points:
                res.append(idx + 1)
        return res

    results = filter_local_max_under_avg(
        formatted_game_log, peaks, player_average)
    a = np.array(results)

    new_list = np.diff(results)

    average = Average(new_list)

    def Find_Breakout(game_log, res):
        try:
            average = np.diff(res)
            average = sum(average) / len(average)
            average = round(average)
            next_game = len(game_log) + 1
            last_breakout = res[-1]
            next_breakout = last_breakout + average
            if(next_game == next_breakout):
                return 1
            else:
                return 0
        except:
            pass

    is_due = Find_Breakout(formatted_game_log, results)

    last_game1 = last_game(clean_array)

    players_list.append([player_one_name, player_team, player_pos, player_average,
                         proj_pass_yds, proj_rush_yds, proj_rec_yds, proj_pass_tds,
                         proj_rush_tds, proj_rec_tds, proj_interceptions, proj_fumbles,
                         proj_rec, proj_tar, proj_rzatt, proj_rztar, proj_touches, is_due, three_game_average, projection, rounded_sharpe,
                         player_std, last_game1])

    print('Player name: ', player_one_name)
    print('Projected Score: ', projection)
    print('Projected PassYds: ', proj_pass_yds)
    print('Projected Rushyds: ', proj_rush_yds)
    print('Projected Recyds: ', proj_rec_yds)
    print('Projected Passtds: ', proj_pass_tds)
    print('Projected Rushtds: ', proj_rush_tds)
    print('Projected Rectds: ', proj_rec_tds)
    print('Projected Int: ', proj_interceptions)
    print('Projected Fum: ', proj_fumbles)
    print('Projected Rec: ', proj_rec)
    print('Projected Tar: ', proj_tar)
    print('Projected RzAtt: ', proj_rzatt)
    print('Projected RzTar: ', proj_rztar)
    print('Projected Touches: ', proj_touches)
    print('Possible Breakout Game: ', is_due)
    print('Average: ', player_average)
    print('Three Game average: ', three_game_average)
    print(' ')

df = pd.DataFrame(players_list, columns=[
    "Name", "Team", "Position", "Average", "Pass Yds","Rush Yds", "Rec Yds", "Pass tds", "Rush tds","Rec tds", "Int", "Fum","Rec","Tar","RzAtt","RzTar","Touches","Breakout", "Three game average", "Projection", "Sharpe Ratio", "STD", "Last Game"])
writer = pd.ExcelWriter('NFL_Player_Stats.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
# print(df)
writer.save()
