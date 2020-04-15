from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
browser = webdriver.Chrome("/ProgramData/chocolatey/bin/chromedriver.exe")

browser.get("https://rotogrinders.com/game-stats?site=fanduel&sport=nba")
# Wait 20 seconds for page to load
timeout = 20
try:
    WebDriverWait(browser, timeout).until(
        EC.visibility_of_element_located((By.CLASS_NAME, 'player-popup')))
except TimeoutException:
    print("Timed out waiting for page to load")
    browser.quit()

browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
button = browser.find_element_by_xpath(
    '//*[@id="game-stats-table"]/div[2]/a[14]')

button.click()
time.sleep(5)
# find_elements_by_xpath returns an array of selenium objects.

players_info = []

players = browser.find_elements_by_tag_name('a.player-popup')

for player in players:
    players_info.append(player.get_attribute('href'))
time.sleep(5)
players_name = [element.split("/", maxsplit=4)[-1] for element in players_info]
browser.quit()
print('Players were scraped successfully')
print(' ')
