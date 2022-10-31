#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y python3-pip xvfb libnss3-dev
# to do "sudo apt install google-chrome-stable", would need to enable the "universe" repository: "sudo add-apt-repository universe"
# instead use this fixed version of google chrome, so that the driver version also matches.
# chrome package and driver both downloaded on 2022-10-31
# chrome download: https://www.google.com/chrome/ --> amd64, version 107.0.5304.87-1
# driver download: https://chromedriver.chromium.org/downloads --> version ChromeDriver 107.0.5304.62
sudo apt install -y /vagrant/google-chrome-stable_amd64.deb

sudo pip install selenium

sudo cp /vagrant/dnsproxy-linux-amd64-v0.46.2 /usr/local/bin/dnsproxy
sudo cp /vagrant/chromedriver /usr/local/bin/chromedriver
