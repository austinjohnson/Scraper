from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

browser = webdriver.Chrome("/ProgramData/chocolatey/bin/chromedriver.exe")

browser.get("https://rotogrinders.com/projected-stats/nfl-defense?site=fanduel")
# Wait 20 seconds for page to load
timeout = 20
try:
    WebDriverWait(browser, timeout).until(
        EC.visibility_of_element_located((By.CLASS_NAME, 'player-popup')))
except TimeoutException:
    print("Timed out waiting for page to load")
    browser.quit()

# find_elements_by_xpath returns an array of selenium objects.

players_info = []

players = browser.find_elements_by_css_selector('div.player')

for player in players:
    link = player.find_element_by_class_name('player-popup')
    players_info.append(link.get_attribute('href'))

players_name = [element.split("/", maxsplit=4)[-1] for element in players_info]

browser.quit()
