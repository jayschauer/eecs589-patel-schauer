#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y python3-pip xvfb libnss3-dev firefox google-chrome-stable

sudo pip install selenium

sudo cp /vagrant/dnsproxy-linux-amd64-v0.46.2 /usr/local/bin/dnsproxy
sudo cp /vagrant/geckodriver-v0.32.0-linux64 /usr/local/bin/geckodriver
sudo cp /vagrant/chromedriver/ /usr/local/bin/chromedriver
