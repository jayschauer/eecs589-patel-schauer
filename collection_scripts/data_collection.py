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
PORT = 853
INTERFACE = 'ens3'

parser = argparse.ArgumentParser()
parser.add_argument('--list', type=str, required=True)
parser.add_argument('--start_line', type=int, required=True)
# end line is exclusive
parser.add_argument('--end_line', type=int, required=True)
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = vars(parser.parse_args())

# load urls from file
urls= []
fname = args['list']
with open(fname) as f:
	lines = f.readlines()
	for line in lines:
		urls.append(line.strip())

start_line = args['start_line']
end_line = args['end_line']
urls = [urls[i] for i in range(start_line, end_line)]

def delete_file(file):
    if os.path.isfile(file):
        os.remove(file)

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
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        driver.execute_cdp_cmd('Network.setCacheDisabled', {'cacheDisabled': True})
        print("Started driver", flush=True)

        # Get url
        try:
            driver.get(url)
            time.sleep(5)
        except TimeoutException as ex:
            print(ex, flush=True)
            delete_file(capture_file)
    except Exception as ex:
        print('Unexpected exception!!')
        print(ex)
        delete_file(capture_file)
    finally:
        # Cleanup driver, tcpdump process
        try:
            os.system(f'sudo pkill -15 tcpdump') # SIGTERM - prevents 'cut short in middle of packet'
        except OSError as error:
            print(error, flush=True)
        driver.quit()
    
    stop = time.time()

    print(process.communicate()[0].decode(), flush=True)
    print(f'Time elapsed: {stop - start}', flush=True)
    print('', flush=True)

    time.sleep(1)
