import pandas as pd
import re
from NBA_Stats import day, month

df = pd.read_excel(
    "/Users/austi/Desktop/PYTHON/webscraping_example/NBA_Stats.xlsx")

df['Name'] = df['Name'].map(lambda x: x.strip(
    '[').strip(']').strip(',').lstrip("'").lstrip(' ').strip("'").replace(',', '').replace("'", ''))

df['Team'] = df['Team'].map(lambda x: x.upper())
df['Opponent'] = df['Opponent'].map(lambda x: x.upper())

print(df)


file_name = "NBA_Stats_"+month+"_"+day+".xlsx"
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.style.set_properties(**{'text-align': 'center'})
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
writer.save()
