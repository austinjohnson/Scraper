from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
import xlsxwriter
import pandas as pd

browser = webdriver.Chrome("/ProgramData/chocolatey/bin/chromedriver.exe") 

browser.get("https://www.rotowire.com/daily/mlb/optimizer.php")

# Wait 20 seconds for page to load
timeout = 20
try:
    WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.CLASS_NAME, 'dplayer-link')))
except TimeoutException:
    print("Timed out waiting for page to load")
    browser.quit()

# find_elements_by_xpath returns an array of selenium objects.

players_info = []

players = browser.find_elements_by_class_name('dplayer-link')

for player in players:
    name = player.find_elements_by_xpath('//*[@id="datatable1566941475317"]/div[2]/div[1]/div/div[2]/div[1]/span/a/text()')
    players_info.append(name.text)
    fp = player.player.find_elements_by_xpath('//*[@id="datatable1566941475317"]/div[2]/div[2]/div/div[6]/div[1]/input')
    players_info.append(fp.get_attribute('value'))

df = pd.DataFrame(players_info, columns = ["Name"])
writer = pd.ExcelWriter('NF_Batters_Names.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
df.style.set_properties(**{'text-align':'center'})
pd.set_option('display.max_colwidth',100)
pd.set_option('display.width', 1000)
#print(df)
writer.save()


print(players_info)



