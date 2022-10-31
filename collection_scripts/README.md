# Data collection 

We use Vagrant + VirtualBox to set up the data collection. 

In order to collect data, perform the following steps:

1. Install VirtualBox.
2. Install Vagrant.
3. Run the command `vagrant up` (should be run from inside the vagrant folder).
4. Log into the VM with the comman `vagrant ssh nodeN`, where N is the ID of the machine (N = 1, in the current configuration). 
5. Run the following command to kick off the experiment: `bash /vagrant/data_collection.sh`

This does the following
- starts data collection from urls in short_list_test
- creates a new folder in data/pcaps and a new subfolder for each iteration
- new folder in logs contains one log file for dnsproxy and the output from the python script for each iteration

# Data processing

`bash preprocess.sh <input dir> <output dir>`
   - reads the pcaps, temporarily writes them to .txt, and saves the label, times, and sizes to a json file in the output directory

# Data Plots

`python3 plot_data.py --dir=<processed data dir> --filename=<output filename>`
   - Plots data from each label on a separate graph and saves it to filename