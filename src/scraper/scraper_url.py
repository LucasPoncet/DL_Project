from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Setup
options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

# URL de la page Explore (modifiable si besoin)
url = "https://www.vivino.com/explore?e=eJzLLbI1VMvNzLM1UMtNrLA1NzUwMFBLrrTNKlBLtvUKiFQrAMqnp9mWJRZlppYk5qjlF6XYJhYnq-UnVdoWFGUmp6qVl0THAhWBKSMIZQyhTCCUOVTOBACInyNS"
driver.get(url)
time.sleep(5)

# Scroll pour forcer le chargement dynamique (JS)
SCROLL_PAUSE_TIME = 2
last_height = driver.execute_script("return document.body.scrollHeight")

for _ in range(3):  # répéter plusieurs fois pour charger plus de vins
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Extraire les URLs des cartes de vin
wine_links = []
wine_cards = driver.find_elements(By.CLASS_NAME, "wineCard__wineCard")

for card in wine_cards:
    try:
        link = card.find_element(By.TAG_NAME, "a").get_attribute("href")
        if link and "/wines/" in link:
            wine_links.append(link)
    except:
        continue

driver.quit()

# Optionnel : sauvegarder dans un fichier
import pandas as pd
pd.DataFrame({"wine_url": wine_links}).to_csv("wine_links.csv", index=False)
