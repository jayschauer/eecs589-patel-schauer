#!/usr/bin/env bash

# command line args
# bash data_collection.sh top-1k-2022-10-31 200 299 50`
url_list=$1
start_line=$2
end_line=$3 # end line inclusive
iterations=$4

# Set up environment
sudo sysctl -w net.core.rmem_max=2500000 # increase buffer size to avoid quic warnings: https://github.com/lucas-clemente/quic-go/wiki/UDP-Receive-Buffer-Size
#sudo cp /etc/resolv.conf ./original_resolv.conf
#sudo cp ./resolv.conf /etc/
sudo timedatectl set-timezone America/Detroit

# Create required directories
d=`date "+%Y-%m-%d_%H-%M-%S"`
mkdir -p ./logs/$d
mkdir -p ./data/pcaps/$d

# Kill any running processes
sudo pkill tcpdump
sudo pkill dnsproxy

# Start dnsproxy server
# 94.140.14.14 is adguard QUIC server: quic://dns.adguard.com
# option --quic-port=853 will listen locally for DNS over QUIC requests on port 853
sudo dnsproxy -l 127.0.0.54 --quic-port=853 -u quic://94.140.14.14 -v > ./logs/$d/dnsproxy_log.txt 2>&1 &
sleep 2
curl www.google.com > /dev/null 2>&1 # send one request so dnsproxy initiates quic tunnel
sleep 2

# Start data collection
N=$iterations
for i in $(seq 1 $N)
do
    echo "Starting iteration $i/$N"
    mkdir -p ./data/pcaps/$d/iteration_$i
	sudo python3 ./data_collection.py --list=$url_list --start_line=$start_line --end_line=$end_line --dir=./data/pcaps/$d/iteration_$i --iter=$i > ./logs/$d/collection_$i.txt 2>&1
	sleep 2
done

# Kill dnsproxy server
sudo pkill dnsproxy
#sudo cp ./original_resolv.conf /etc/resolv.conf
