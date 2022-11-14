import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--label', type=str, required=True)
args = vars(parser.parse_args())

filename = args['file']
label = args['label']

# Create dictionary to store label and time/size data
data = {'label': label, 'time': [], 'size': []}
LOCAL_IP = '192.168.122.215'

# Open file and read data
with open(filename, 'r') as infile:
    for line in infile:
        line = line.split()
        if line[2] == LOCAL_IP:  # outgoing packet
            sign = 1
        else:  # incoming packet
            sign = -1
        data['time'].append(float(line[1]))
        data['size'].append(sign * int(line[6]))

# Create the output url directory if it doesn't exist
output_dir = os.path.join(args['dir'], label)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

i = len(os.listdir(output_dir))
# Write the processed data to a json file
output_name = os.path.join(output_dir, f'{i}.json')
with open(output_name, 'w') as outfile:
    json.dump(data, outfile)
