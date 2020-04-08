from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
import statistics

browser = webdriver.Chrome("/ProgramData/chocolatey/bin/chromedriver.exe") 

browser = webdriver.Chrome("/ProgramData/chocolatey/bin/chromedriver.exe")

browser.get("https://www.linestarapp.com/LiveScoring/Sport/PGA/Site/FanDuel/PID/181")
total_percentage_dif = []
res = 0
while res < 37:
    players_score = []
    players_info = []
    date = browser.find_element_by_xpath('//*[@id="dnn_ctr754_ModuleContent"]/div/div/div[2]/div[3]/a')
    date.click()

    browser.implicitly_wait(3) # seconds
    # Wait 20 seconds for page to load
    timeout = 20
    try:
        WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.CLASS_NAME, 'liveScoringTableRow')))
    except TimeoutException:
        print("Timed out waiting for page to load")
        browser.quit()
    
    link = browser.find_element_by_xpath('//*[@id="dnn_ctr754_ModuleContent"]/div/div/div[1]/div/span')
    players_info.append(link.text)
    players = browser.find_elements_by_class_name('liveScoringTableRow')
   
    for player in players:
        link1 = player.find_element_by_class_name('liveScoringTableCell')
        players_info.append(link1.text)
        link2 = player.find_element_by_class_name('liveScoringTableCell.lstcScored')
        players_score.append(link2.text)
    res = res + 1

    average = list(map(float, players_score)) 
    average1 = statistics.mean(average)
    top_score = average[0]
   
    percentage_dif = ((top_score - average1) / average1) + 1
    total_percentage_dif.append(percentage_dif)
    print(total_percentage_dif)
average_tot_perc_dif = statistics.mean(total_percentage_dif)
print(round(average_tot_perc_dif,2))
    





