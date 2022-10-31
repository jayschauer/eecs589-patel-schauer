import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException
import sys

start = time.time()
INTERVAL_TIME = 1 #Interval time between queries
urls = []
ct = 0

# fname = "/vagrant/short_list_1500"
fname = "/vagrant/short_list_2"
with open(fname) as f:
	lines = f.readlines()
	for line in lines:
		urls.append(line.strip())

url = urls[int(sys.argv[1])]
print(url)

options = webdriver.FirefoxOptions()
options.add_argument("--headless")
service = Service(executable_path='/usr/local/bin/geckodriver')
driver = webdriver.Firefox(service=service, options=options)
print("Started Firefox driver")
url = 'http://' + url
try:
	driver.get(url)
except TimeoutException as ex:
	print(ex)
driver.quit()
# display.stop()
stop = time.time()
print("Time taken:" + str(stop - start))
time.sleep(INTERVAL_TIME)
