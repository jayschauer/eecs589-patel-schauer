# Data collection 

We use Vagrant + VirtualBox to set up the data collection. 

In order to collect data, perform the following steps:

1. Install VirtualBox.
2. Install Vagrant.
3. Run the command `vagrant up` (should be run from inside the vagrant folder).
4. Log into the VM with the command `vagrant ssh node1`.
5. Run the following command to kick off the experiment: `bash /vagrant/data_collection.sh`.

This does the following
- starts data collection from urls in short_list_test
- creates a new folder in data/pcaps and a new subfolder for each iteration
- new folder in logs contains one log file for dnsproxy and the output from the python script for each iteration

# Data processing
1. `vagrant ssh node1`
2. Make sure tshark is installed: `sudo apt-get install tshark`
3. `bash /vapgrant/preprocess.sh <input dir> <output dir>`
   - reads the pcaps, temporarily writes them to .txt, and saves the label, times, and sizes to a json file in the output directory
   - e.g. `bash /vapgrant/preprocess.sh /vagrant/data/pcaps/31-10-22-145734 /vagrant/data/processed`

# Data Plots
1. `vagrant ssh node1`
2. Make sure numpy and matplotlib are installed: `sudo pip install numpy matplotlib`
3. `python3 /vagrant/analysis/plot_data.py --dir=<processed data dir> --filename=<output filename>`
   - Plots data from each label on a separate graph and saves it to filename