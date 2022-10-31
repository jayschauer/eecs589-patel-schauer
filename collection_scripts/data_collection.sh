#!/usr/bin/env bash

# Set up environment
sudo sysctl -w net.core.rmem_max=2500000 # increase buffer size to avoid quic warnings: https://github.com/lucas-clemente/quic-go/wiki/UDP-Receive-Buffer-Size
sudo cp /etc/resolv.conf /vagrant/original_resolv.conf
sudo cp /vagrant/resolv.conf /etc/
sudo timedatectl set-timezone America/Detroit

# Create required directories
d=`date "+%d-%m-%y-%H%M%S"`
mkdir -p /vagrant/logs/$d
mkdir -p /vagrant/data/pcaps/$d

# Kill any running processes
sudo pkill tcpdump
sudo pkill dnsproxy

# Start dnsproxy server
# 94.140.14.14 is adguard QUIC server: quic://dns.adguard.com
# option --quic-port=853 will listen locally for DNS over QUIC requests on port 853
sudo dnsproxy -l 127.0.0.54 --quic-port=853 -u quic://94.140.14.14 -v > /vagrant/logs/$d/dnsproxy_log.txt 2>&1 &
sleep 2
curl www.google.com > /dev/null 2>&1 # send one requests so dnsproxy initiates quic tunnel
sleep 2

# Start data collection
N=1 # 100 samples per domain
for i in $(seq 1 $N)
do
    echo "Starting iteration $i/$N"
    mkdir -p /vagrant/data/pcaps/$d/iteration_$i
	sudo python3 /vagrant/data_collection.py --dir=/vagrant/data/pcaps/$d/iteration_$i --iter=$i > /vagrant/logs/$d/collection_$i.txt 2>&1
	sleep 2
done

# Kill dnsproxy server
sudo pkill dnsproxy
sudo cp /vagrant/original_resolv.conf /etc/resolv.conf