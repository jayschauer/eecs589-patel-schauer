import os
import glob
import json
import argparse
import shutil
import pickle


def get_label_to_files_dict(input_dir):
    '''
    Gets dictionary mapping labels to a list of data files with that label.
    
    input_dir: Directory containing the .json data files. May have subdirectories.

    Returns: dictionary with label as key and a list of filenames as value.
    '''
    data = {}
    path = os.path.join(input_dir, '**/*.json')
    for filename in glob.glob(path, recursive=True):
        # get the label from the json
        with open(filename, 'r') as infile:
            file_data = json.load(infile)

        label = file_data['label']
        if label not in data:
            data[label] = []

        data[label].append(filename)  # add filename to list for label

    return data


def copy_to_new_dir(label_dict, output_dir):
    '''
    Copy data files for a label into a single directory. Useful for merging
    subdirectories that all have the same filenames for a label.

    label_dict: dictionary with label as key and a list of filenames as value.
    output_dir: directory to copy files to. A single subdirectory for each label
        will be created inside output_dir.
    '''
    for label in label_dict:
        # Make a new directory for label in output_dir
        path = os.path.join(output_dir, label)
        os.makedirs(path)

        # Copy all files to new directory
        for i, filename in enumerate(label_dict[label]):
            output_file = os.path.join(path, f'{i}.json')
            shutil.copyfile(filename, output_file)


def merge_directories(args):
    '''
    Copies files for a label from multiple locations to single directory.

    args: parsed command line arguments.
    '''
    input_dir = args['input_dir']
    output_dir = args['output_dir']

    data = get_label_to_files_dict(input_dir)
    copy_to_new_dir(data, output_dir)


def get_label_dict(path):
    '''
    Get a dictionary mapping a label to its position in a list.

    path: path to url list file.

    Returns: a dictionary with the label as key and index as value.
    '''
    # Get labels from file
    label_dict = {}
    with open(path) as label_file:
        lines = label_file.readlines()
        for i, line in enumerate(lines):
            label_dict[line.strip()] = i

    return label_dict


def pickle_data(args):
    '''
    Loads json data files and saves them as a pickle.

    args: command line arguments

    Returns: None (saves a dictionary as a pickle with two keys, 'data' and 'labels').
    '''
    input_dir = args['input_dir']
    filename = args['filename']
    label_filename = args['label_file']
    num_samples = args['num_samples']

    data = []
    labels = []
    label_dict = get_label_dict(label_filename)
    label_to_files_dict = get_label_to_files_dict(input_dir)

    # load data from files in dict
    for label in label_to_files_dict:
        files = label_to_files_dict[label]
        # if fixed number of samples is requested, use the first <num_samples> files
        if num_samples is not None and num_samples > 0:
            files = files[:num_samples]  # will get first min(num_samples, len(files)) elements

        for file in files:
            with open(file, 'r') as json_file:
                file_data = json.load(json_file)

            label = file_data['label']
            times = file_data['time']
            sizes = file_data['size']
            directions = file_data['direction']

            series = dict(zip(times, zip(sizes, directions)))  # store as dictionary of (time: (size, direction)) pairs
            data.append(series)
            labels.append(label_dict.get(label, -1))  # index of url in list, -1 if not present

    # Save data as pickle
    output_path = os.path.join(input_dir, filename)
    with open(output_path, 'wb') as pickle_file:
        d = {'data': data, 'labels': labels}
        pickle.dump(d, pickle_file)


def load_data(path):
    '''
    Loads data from the pickle file.

    path: path to the file to load.

    Returns: data, labels as lists.
    '''
    data, labels = [], []
    with open(path, 'rb') as data_file:
        d = pickle.load(data_file)
        data = d['data']
        labels = d['labels']

    return data, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='execution mode', dest='mode')

    merge_parser = subparsers.add_parser('merge')
    merge_parser.add_argument('--input_dir', type=str, required=True,
                              help='directory containing subdirectories to merge')
    merge_parser.add_argument('--output_dir', type=str, required=True,
                              help='directory to copy merged subdirectories to')
    merge_parser.set_defaults(func=merge_directories)

    pickle_parser = subparsers.add_parser('pickle')
    pickle_parser.add_argument('--input_dir', type=str, required=True, help='directory containing data files')
    pickle_parser.add_argument('--filename', type=str, required=True, help='file in INPUT_DIR to write to')
    pickle_parser.add_argument('--label_file', type=str, required=True, help='file containing list of urls')
    pickle_parser.add_argument('--num_samples', type=int, required=False,
                               help='if specified, the number of samples for each label to include (useful for making '
                                    'limited dataset)')
    pickle_parser.set_defaults(func=pickle_data)

    # Get args and call the function associated with the execution mode
    args = vars(parser.parse_args())
    args['func'](args)
