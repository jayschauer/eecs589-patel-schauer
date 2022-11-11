import os
import glob
import json
import pandas as pd
import pickle

import sktime.datatypes as datatypes
from sktime.transformations.panel.padder import PaddingTransformer
# from sktime.classification.interval_based import SupervisedTimeSeriesForest
# from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Loads the url file to use indices as labels
def get_label_dict(path):
    label_dict = {}
    with open(path) as label_file:
        lines = label_file.readlines()
        for i, line in enumerate(lines):
            label_dict[line.strip()] = i

# Loads the data from the specified path. If it's a pickle file it will load that,
# otherwise load data from json files in directory and save as pickle
# Returns:  data - list of samples where each sample is a dictionary of (time: size) values
#           labels - list of labels where the label is the value of the url from the dictionary
def load_data(data_path, label_dict):
    data = []
    labels = []

    # If it's a pickled file, unpickle it
    if os.path.isfile(data_path):
        print('Loading from pickle file')
        with open(data_path, 'rb') as data_file:
            d = pickle.load(data_file)
            data = d['data']
            labels = d['labels']

    else:
        print('Loading from json files')
        path = os.path.join(data_path, '**/*.json')
        for filename in glob.glob(path, recursive=True):
            with open(filename, 'r') as infile:
                file_data = json.load(infile)

                label = file_data['label']
                times = file_data['time']
                sizes = file_data['size']
                
                series = dict(zip(times, sizes))
                data.append(series)
                labels.append(label_dict.get(label, -1))  # index of url in list, -1 if not present

        # Save data as pickle for future use
        pickle_path = os.path.join(data_path, 'data.pkl')
        with open(pickle_path, 'wb') as pickle_file:
            d = {'data': data, 'labels': labels}
            pickle.dump(d, pickle_file)

    
    return data, labels

# Gets the list of urls
def get_label_dict(path):
    # Get labels from file
    label_dict = {}
    with open(path) as label_file:
        lines = label_file.readlines()
        for i, line in enumerate(lines):
            label_dict[line.strip()] = i

    return label_dict

# Converts the data list returned by load_data to a dataframe format recognized by sktime classifiers
# Returns:  multi-index panel of mtype 'pd-multiindex'
def make_dataframe(data):
    cols = ['instances', 'timepoints', 'packet_size']
    
    # make a list of dataframes where each frame has rows (index, time, size) for row index in data
    Xlist = [
        pd.DataFrame(
            [ [i, time, size] for time, size in series.items() ],
            columns=cols
        ) for i, series in enumerate(data)
    ]
    
    # convert to sktime panel mtype
    X = datatypes.convert_to(Xlist, to_type='pd-multiindex')
    
    return X

# Loads data and trains classifier
def classify():
    data_dir = './collection_scripts/data/processed_half/data.pkl'
    url_file = './top-1k-curated'

    print('Loading data...')
    data, labels = load_data(data_dir, url_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, random_state=589, test_size=0.25, shuffle=True
    )
    X_train, X_test = make_dataframe(X_train), make_dataframe(X_test)
    y_train = pd.Series(y_train)

    print('Starting training...')
    clf = PaddingTransformer() * RocketClassifier()
    clf.fit(X_train, y_train)

    print('Making predictions...')
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    print(f'Accuracy: {acc}')


classify()
