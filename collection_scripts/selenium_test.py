import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')

service = Service(executable_path='/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(30)
driver.execute_cdp_cmd('Network.setCacheDisabled', {'cacheDisabled': True})

start = time.time()
driver.get('http://google.com')
elapsed = time.time() - start
print('elapsed 1: ' + str(elapsed))
driver.get('http://youtube.com')
elapsed = time.time() - start
print('elapsed 2: ' + str(elapsed))
driver.get('https://nghttp2.org/httpbin/delay/5')
elapsed = time.time() - start
print('elapsed 3: ' + str(elapsed))
driver.get('https://nghttp2.org/httpbin/delay/10')
elapsed = time.time() - start
print('elapsed 4: ' + str(elapsed))
driver.quit()