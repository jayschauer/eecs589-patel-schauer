#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y python3-pip xvfb libnss3-dev firefox

sudo pip install selenium

sudo cp /vagrant/dnsproxy-linux-amd64-v0.46.2 /usr/local/bin/dnsproxy
sudo cp /vagrant/geckodriver-v0.32.0-linux64 /usr/local/bin/geckodriver
sudo nohup /usr/local/bin/dnsproxy -l 127.0.0.54 -u quic://dns.adguard.com -b 1.1.1.1:53 & #-v > /vagrant/dnsproxy.out 2>&1 &
sudo cp /vagrant/resolv.conf /etc/

sudo timedatectl set-timezone America/Detroit

mkdir -p /vagrant/pcaps/$1
mkdir -p /vagrant/pcaps2/$1
mkdir -p /vagrant/logs/$1
