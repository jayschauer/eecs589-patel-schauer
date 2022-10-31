import argparse
import os
import subprocess
import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException

# Constants
INTERVAL_TIME = 5  # Interval time between queries
PORT = 853
INTERFACE = 'enp0s3'

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = vars(parser.parse_args())

# load urls from file
urls= []
fname = "/vagrant/repeat_test"
with open(fname) as f:
	lines = f.readlines()
	for line in lines:
		urls.append(line.strip())

def load_once():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    
    service = Service(executable_path='/usr/local/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    driver.execute_cdp_cmd('Network.setCacheDisabled', {'cacheDisabled': True})
    try:
        driver.get('http://google.com')
        time.sleep(INTERVAL_TIME)
    except TimeoutException as ex:
        print(ex, flush=True)

    driver.quit()

load_once()

for i, base_url in enumerate(urls):
    start = time.time()
    url = 'http://' + base_url
    print(f'URL: {url}', flush=True)

    # Start capture via tcpdump
    capture_file = os.path.join(args['dir'], f'{base_url}.pcap')
    print(capture_file)
    cmd = f'sudo tcpdump -i {INTERFACE} port {PORT} -w {capture_file}'.split()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Create chrome driver instance
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    
    service = Service(executable_path='/usr/local/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    driver.execute_cdp_cmd('Network.setCacheDisabled', {'cacheDisabled': True})
    print("Started driver", flush=True)
    
    # Get url
    try:
        driver.get(url)
        time.sleep(INTERVAL_TIME)
    except TimeoutException as ex:
        print(ex, flush=True)

    # Cleanup driver and tcpdump process
    driver.quit()
    try:
        os.system(f'sudo pkill -15 tcpdump') # SIGTERM - prevents 'cut short in middle of packet'
    except OSError as error:
        print(error, flush=True)

    stop = time.time()

    print(process.communicate()[0].decode(), flush=True)
    print(f'Time elapsed: {stop - start}', flush=True)
    print('', flush=True)

    time.sleep(1)