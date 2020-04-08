from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import statistics
import requests
import json
import numpy as np
import statistics
import pandas as pd
import xlsxwriter
import re

browser = webdriver.Chrome("/ProgramData/chocolatey/bin/chromedriver.exe")
month = input("Enter the month(in number format): ")
day = input("Enter the day: ")

url = "http://rotoguru1.com/cgi-bin/hyday.pl?game=fd&mon=" + \
    month+"&day="+day+"&year=2019"
browser.get(url)

table_rows = browser.find_element_by_xpath(
    '/html/body/table/tbody/tr/td[3]/table[4]').find_element_by_tag_name('tbody').find_elements_by_tag_name('tr')

players = []

for row in table_rows:
    cells = row.find_elements_by_tag_name('td')
    if len(cells) < 2:
        continue
    else:
        pos = cells[0].text
        name = cells[1].text
        name_1 = name.replace('^', '')
        tmp = name_1
        tmp_2 = tmp.split(',')
        tmp_2.reverse()
        "".join(tmp_2)
        fpts = cells[2].text
        salary = cells[3].text
        salary_2 = salary.replace('$', '')
        salary_3 = salary_2.replace(',', '')
        team = cells[4].text
        opp = cells[5].text
        opp_2 = opp.replace('v', '')
        opp_3 = opp_2.replace('@', '')
        minutes = cells[7].text
        players.append([pos, tmp_2, fpts, salary_3, team, opp_3, minutes])


def clean_list(list):
    for item in list:
        if ['Guards', ['FD Points'], 'Salary', 'Team', 'Opp.', 'Score', '  Stats'] in list:
            list.remove(item)
        elif ['Forwards', ['FD Points'], 'Salary', 'Team', 'Opp.', 'Score', '  Stats'] in list:
            list.remove(item)
        elif ['Centers', ['FD Points'], 'Salary', 'Team', 'Opp.', 'Score', '  Stats'] in list:
            list.remove(item)
        else:
            continue
    return list


clean_player_list = clean_list(players)

df = pd.DataFrame(clean_player_list, columns=[
    "Position", "Name", "FPTS", "Salary", "Team", "Opponent", "Minutes"])
writer = pd.ExcelWriter('NBA_Stats.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
print('Players were gathered successfully!')
print(clean_player_list)
writer.save
