#/usr/bin/bash
# while true; do
d=`date "+%d-%m-%y-%H%M%S"`
mkdir -p /vagrant/pcaps/$1
mkdir -p /vagrant/pcaps/$1/$d
sudo pkill tcpdump
sleep 3
# for i in {0..1499}
for i in {0..1}
do
	echo $i
    # safe to just listen to port 853 since nothing actually uses this port.
	sudo /usr/sbin/tcpdump -i enp0s3 port 853 -w /vagrant/pcaps/$1/$d/$i.pcap &
	sleep 2
	python3 /vagrant/firefox_driver.py $i
	sleep 2
	sudo pkill tcpdump
	sleep 2
done
# sleep 300
# done
