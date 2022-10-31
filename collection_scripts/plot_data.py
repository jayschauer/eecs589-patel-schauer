import argparse
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_color(i):
    colors = list(mcolors.TABLEAU_COLORS.values())
    return colors[i % len(colors)]

def load_data(data_dir):
    data = {}

    path = os.path.join(data_dir, '**/*.json')
    for filename in glob.glob(path, recursive=True):
        with open(filename, 'r') as infile:
            file_data = json.load(infile)

        label = file_data['label']
        times = file_data['time']
        sizes = file_data['size']

        if label not in data:
            data[label] = { 'times': [], 'sizes': [] }
        
        data[label]['times'].append(times)
        data[label]['sizes'].append(sizes)

    return data

def plot_data(data, output='./test.png'):
    domains = data.keys()
    fig, ax = plt.subplots(1, len(domains), figsize=(22, 5))

    for i, domain in enumerate(domains):
        time_list = data[domain]['times']
        size_list = data[domain]['sizes']

        color = color=get_color(i)
        for j, (time, size) in enumerate(zip(time_list, size_list)):
            ax[i].scatter(time, size, color=get_color(j), label=j)
            ax[i].set_title(domain)
            ax[i].set_xlabel('time (s)')
            ax[i].set_ylabel('packet size (bytes)')
            ax[i].legend()

    plt.savefig(output)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--filename', type=str)
args = vars(parser.parse_args())

data = load_data(args['dir'])
plot_data(data, args['filename'])