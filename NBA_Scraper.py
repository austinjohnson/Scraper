import requests
import json
import numpy as np
import statistics as st
import pandas as pd
import xlsxwriter
from webscraping_example_NBA import players_name
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from scipy.signal import find_peaks, argrelextrema
import time

start_time = time.time()


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
    minutes = []
    usg = []

    points = []
    assists = []
    blocks = []
    rebounds = []
    steals = []
    turnovers = []

    while i < games_played:
        try:
            fpts.append(cont['stats']['this-season'][i]['fpts']['2'])
            salary.append(cont['stats']['this-season'][i]['salary']['2'])
            minutes.append(cont['stats']['this-season'][i]['min'])
            usg.append(cont['stats']['this-season'][i]['usg'])
            points.append(cont['stats']['this-season'][i]['pts'])
            assists.append(cont['stats']['this-season'][i]['ast'])
            blocks.append(cont['stats']['this-season'][i]['blk'])
            rebounds.append(cont['stats']['this-season'][i]['reb'])
            steals.append(cont['stats']['this-season'][i]['stl'])
            turnovers.append(cont['stats']['this-season'][i]['to'])
        except Exception:
            pass
        i += 1

    for n, i in enumerate(fpts):
        if i == '-':
            fpts[n] = 0

    salary = list(filter(None, salary))
    salary = list(map(float, salary))
    salary = list(filter(lambda a: a != 0, salary))

    usg = list(filter(None, usg))
    usg = list(map(float, usg))
    usg = list(filter(lambda a: a != 0, usg))

    fpts = list(filter(None, fpts))
    fpts = list(map(float, fpts))
    fpts = list(filter(lambda a: a != 0, fpts))

    minutes = list(filter(None, minutes))
    minutes = list(map(float, minutes))
    minutes = list(filter(lambda a: a != 0, minutes))
    # return ROI for every player
    roi_calc = [int(fpts) / int(salary) for fpts, salary in zip(fpts, salary)]
    value_calc = [int(fpts) / (int(salary)/1000)
                  for fpts, salary in zip(fpts, salary)]

    def Remove_Zeros(lst):
        try:
            lst = list(filter(lambda a: a != 0, lst))
            return lst
        except:
            return 0

    clean_points = Remove_Zeros(points)
    clean_assists = Remove_Zeros(assists)
    clean_blocks = Remove_Zeros(blocks)
    clean_rebounds = Remove_Zeros(rebounds)
    clean_steals = Remove_Zeros(steals)
    clean_turnovers = Remove_Zeros(turnovers)

    def Complete_List(lst, length):
        try:
            while len(lst) < length:
                avg = sum(lst) / len(lst)
                avg = round(avg, 2)
                sd = st.stdev(lst)
                sd = round(sd, 2)
                ceil = avg + sd
                ceil = round(ceil, 2)
                floor = avg - sd
                floor = round(floor, 2)
                lst.append(uniform(floor, ceil))
            return lst
        except:
            pass

    game_length = 42
    complete_fpts = Complete_List(fpts, game_length)
    complete_salary = Complete_List(salary, game_length)
    complete_minutes = Complete_List(minutes, game_length)
    complete_usg = Complete_List(usg, game_length)
    complete_points = Complete_List(clean_points, game_length)
    complete_assists = Complete_List(clean_assists, game_length)
    complete_blocks = Complete_List(clean_blocks, game_length)
    complete_rebounds = Complete_List(clean_rebounds, game_length)
    complete_steals = Complete_List(clean_steals, game_length)
    complete_turnovers = Complete_List(clean_turnovers, game_length)

    def value_six_or_above(lst):
        new_lst = []
        try:
            for x in lst:
                if x >= 6:
                    new_lst.append(1)
            count = len(new_lst)
            count = count / len(lst)
            count = round(count, 3)
        except:
            pass

        return count

    boom = value_six_or_above(value_calc)

    def fantasy_points_per_minute(lst1, lst2):
        lst1 = filter(lambda x: x > 0, lst1)
        lst2 = filter(lambda x: x > 0, lst2)
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

    clean_minutes = no_zeros_array(complete_minutes)
    # removes all zeros from array
    clean_array = no_zeros_array(complete_fpts)

    def Reverse_array(lst):
        try:
            lst.reverse()
            return lst
        except:
            return 0

    clean_array = Reverse_array(clean_array)

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
            lst = list(filter(None, lst))
            return sum(lst) / len(lst)
        except:
            pass

    def Last_Five(lst):
        try:
            new_lst = lst[-5:]
            five_game_avg = sum(new_lst) / len(new_lst)
            return round(five_game_avg, 2)
        except:
            pass

    five_game_average = Last_Five(clean_array)

    max_deviation = calc_deviation(player_ceil, player_average)
    min_deviation = calc_deviation(player_floor, player_average)

    player_average_min = Average(clean_minutes)

    def Round_Player_minutes(lst):
        try:
            return round(lst, 2)
        except:
            pass

    rounded_player_average_min = Round_Player_minutes(player_average_min)

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

    # 2 dimensional array for fpts
    two_d_lst = create_2d_lst(clean_array_minus_outliers)
    two_d_lst_min = create_2d_lst(complete_minutes)
    two_d_lst_usg = create_2d_lst(complete_usg)
    two_d_lst_points = create_2d_lst(complete_points)
    two_d_lst_assists = create_2d_lst(complete_assists)
    two_d_lst_blocks = create_2d_lst(complete_blocks)
    two_d_lst_rebounds = create_2d_lst(complete_rebounds)
    two_d_lst_steals = create_2d_lst(complete_steals)
    two_d_lst_turnovers = create_2d_lst(complete_turnovers)

    def Predictor(data):
        try:
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
            alpha = 0.001  # learning rate
            iters = 2000  # iterations

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
        except:
            return 0

    projection = Predictor(two_d_lst)
    minutes_proj = Predictor(two_d_lst_min)
    usage_proj = Predictor(two_d_lst_usg)
    points_proj = Predictor(two_d_lst_points)
    assists_proj = Predictor(two_d_lst_assists)
    blocks_proj = Predictor(two_d_lst_blocks)
    rebounds_proj = Predictor(two_d_lst_rebounds)
    steals_proj = Predictor(two_d_lst_steals)
    turnovers_proj = Predictor(two_d_lst_turnovers)

    def last_game(lst):
        try:
            return lst[-1]
        except:
            pass

    last_game1 = last_game(clean_array)

    fppm = fantasy_points_per_minute(fpts, minutes)

    average_fppm = Average(fppm)

    def Round_Player_FPPM(x):
        try:
            return round(x, 2)
        except:
            pass

    try:
        rounded_clean_array = [round(x) for x in clean_array]
        formatted_game_log = np.array(rounded_clean_array)

        y = formatted_game_log
        y = np.array(y)
        x = (range(0, (len(formatted_game_log))))
        x = np.array(x)
        z = np.polyfit(x, y, 1)
        z = z[0]
    except:
        pass

    peaks, _ = find_peaks(y)

    peaks = list(peaks)

    def filter_local_max_under_avg(game_log, peaks, average_points):
        try:
            res = list()
            for idx in peaks:
                if game_log[idx] > average_points:
                    res.append(idx + 1)
            return res
        except:
            pass

    results = filter_local_max_under_avg(
        formatted_game_log, peaks, player_average)

    def Find_Breakout(game_log, res):
        try:
            average = np.diff(res)
            average = sum(average) / len(average)
            average = round(average)
            next_game = len(game_log) + 1
            last_breakout = res[-1]
            next_breakout_ceil = last_breakout + (average + 1)
            next_breakout_floor = last_breakout + (average - 1)
            if(next_breakout_ceil >= next_game and next_game >= next_breakout_floor):
                return 1
            else:
                return 0
        except:
            pass

    def Calculated_Projection_Based_Off_Stats(points, assists, blocks, rebounds, steals, turnovers):
        try:
            points = points
            assists = assists * 1.5
            blocks = blocks * 3
            rebounds = rebounds * 1.2
            steals = steals * 3
            turnovers = turnovers * -1
            FPTS = (points + assists + blocks + rebounds + steals + turnovers)
            FPTS = round(FPTS, 2)
            return FPTS
        except:
            pass

    points_off_stats = Calculated_Projection_Based_Off_Stats(
        points_proj, assists_proj, blocks_proj, rebounds_proj, steals_proj, turnovers_proj)

    up_or_down = z
    up_or_down = round(up_or_down, 3)

    def Fantasy_Points_Per_Minute(arg1, arg2):
        try:
            points = arg1*arg2
            points = round(points, 2)
            return points
        except:
            pass

    try:
        projected_fppm = Fantasy_Points_Per_Minute(average_fppm, minutes_proj)
        projected_points_from_fppm = round(projected_fppm, 2)
    except:
        pass

    is_due = Find_Breakout(formatted_game_log, results)

    value = [int(fpts) / (int(salary)/1000)
             for fpts, salary in zip(fpts, salary)]

    try:
        above_six = sum(i > 6 for i in value) / len(fpts)
        above_seven = sum(i > 7 for i in value) / len(fpts)
        above_eight = sum(i > 8 for i in value) / len(fpts)
        above_nine = sum(i > 9 for i in value) / len(fpts)
        above_ten = sum(i > 10 for i in value) / len(fpts)
        above_eleven = sum(i > 11 for i in value) / len(fpts)
        above_twelve = sum(i > 12 for i in value) / len(fpts)
    except:
        pass

    rounded_fppm = Round_Player_FPPM(average_fppm)
    game_log = clean_array
    players_list.append([player_one_name, player_team,
                         player_pos, player_average, five_game_average,
                         points_proj, assists_proj,
                         blocks_proj, rebounds_proj,
                         steals_proj, turnovers_proj, above_six, above_seven, above_eight, above_nine, above_ten, above_eleven, above_twelve,
                         rounded_fppm,
                         projection, points_off_stats,
                         player_std,
                         rounded_player_average_min,
                         minutes_proj, usage_proj, last_game1, game_log])

    print('Player Name: ', player_one_name)
    print('Player Average: ', player_average)
    print('Player 5 Game Average: ', five_game_average)
    print(' ')

df = pd.DataFrame(players_list, columns=[
    "Name", "Team", "Position",
    "Average", "Five game average", "Projected Points",
    "Projected Assists", "Projected Blocks",
    "Projected Rebounds", "Projected Steals",
    "Projected Turnovers", "Value Above 6","Value Above 7","Value Above 8","Value Above 9","Value Above 10","Value Above 11","Value Above 12",
    "FPPM",
    "Projection", "Projection based on Stats", "STD",
    "Average Minutes", "Projected Minutes", "Usage Projection",
    "Last Game", "Games"])
writer = pd.ExcelWriter('NBA_Player_Stats.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
# print(df)
writer.save()
print(round(((time.time() - start_time)/60)), "minute run time")
