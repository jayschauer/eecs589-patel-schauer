import argparse
import os
import pandas as pd
import pickle
import sys

import sktime.datatypes as datatypes
from sktime.transformations.panel.padder import PaddingTransformer
# from sktime.classification.interval_based import SupervisedTimeSeriesForest
# from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../collection_scripts')
sys.path.append(collection_path)
from dataset_utils import load_data

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
def classify(args):
    print('Loading data...')
    data, labels = load_data(args['data'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, random_state=589, test_size=args['test_size'], shuffle=True
    )
    X_train, X_test = make_dataframe(X_train), make_dataframe(X_test)
    y_train = pd.Series(y_train)

    # get maximum series length for padder
    max_length = max(map(lambda sample: len(sample), data))

    print('Starting training...')
    clf = PaddingTransformer(pad_length=max_length) * RocketClassifier()
    clf.fit(X_train, y_train)

    print('Making predictions...')
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    print(f'Accuracy: {acc}')

    predictions_file = args['filename']
    with open(predictions_file, 'wb') as pred_file:
        d = {'pred': pred, 'gt': y_test}
        pickle.dump(d, pred_file)


if __name__=='__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data', type=str, required=True, help='Pickle file containing the data')
    parser.add_argument('--filename', type=str, required=False, default='predictions.pkl', help='Path to save predictions to. Default is predictions.pkl in current directory')
    parser.add_argument('--test_size', type=float, required=False, default=0.25, help='Size of test split. Default is 0.25')
    args = vars(parser.parse_args())

    classify(args)