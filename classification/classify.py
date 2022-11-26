import argparse
import os
import pandas as pd
import pickle
import sys
import numpy as np

import sktime.datatypes as datatypes
from sktime.transformations.panel.padder import PaddingTransformer
# sktime classifiers are lazy loaded when used to reduce overhead

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../collection_scripts')
sys.path.append(collection_path)
from dataset_utils import load_data

CLASSIFIER_TYPES = ['rocket', 'minirocket', 'knn', 'summary', 'catch22']


def make_dataframe(data):
    """
    Convert the data from the list of time/series representation to a multiindex dataframe.

    data: list of samples where each sample is a dictionary of time: size pairs

    Returns: pd-multiindex dataframe where first index is instance and second index is time point.
    """
    cols = ['time', 'packet_size']

    # make a list of dataframes where each frame has rows (time, size) for row index in data
    Xlist = [
        pd.DataFrame(
            [[time, datum[0]] for time, datum in series.items()],
            columns=cols
        ) for series in data
    ]

    # convert to sktime panel
    # basically does X = pd.concat(obj, axis=0, keys=range(len(Xlist)), names=["instances", "timepoints"])
    X = datatypes.convert_to(Xlist, to_type='pd-multiindex')

    # pd_multi-index support not great in skitime code, even though it is the recommended way to format data
    return X


def make_directional_dataframe(data):
    """
    Same as above, but uses the direction. Outgoing is 1, incoming is -1
    """
    cols = ['time', 'packet_size', 'direction']
    Xlist = [
        pd.DataFrame(
            [[time, datum[0], datum[1]] for time, datum in series.items()],
            columns=cols
        ) for series in data
    ]
    return datatypes.convert_to(Xlist, to_type='pd-multiindex')


def get_classifier(method, max_length):
    '''
    Selects classifier to use based on command line arguments.

    method: method from command line arguments
    max_length: length of longest sequence in input data (used to construct padder)

    Returns: classification pipeline
    '''
    padder = PaddingTransformer(pad_length=max_length)

    if method == 'rocket':
        print('Using ROCKET classifier.')
        from sktime.classification.kernel_based import RocketClassifier
        clf = padder * RocketClassifier()

    elif method == 'minirocket':
        print('Using MINIROCKET transformer with LogisticRegression classifier.')
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        from sklearn.linear_model import LogisticRegressionCV
        clf = padder * MiniRocketMultivariate(n_jobs=-1) * StandardScaler() \
            * LogisticRegressionCV(n_jobs=-1, multi_class='ovr', tol=0.005, verbose=1)

    elif method == 'knn':
        print('Using KNN time series classifier.')
        from sktime.classification.distance_based import \
            KNeighborsTimeSeriesClassifier
        from sktime.alignment.dtw_python import AlignerDTW
        from sktime.dists_kernels.compose_from_align import DistFromAligner

        dtw_dist = DistFromAligner(AlignerDTW())
        clf = KNeighborsTimeSeriesClassifier(distance=dtw_dist)

    elif method == 'summary':
        print('Using Summary classifier.')
        from sktime.classification.feature_based import SummaryClassifier
        from sklearn.ensemble import RandomForestClassifier
        clf = padder * SummaryClassifier(estimator=RandomForestClassifier(n_estimators=5))

    elif method == 'catch22':
        print('Using Catch22 classifier.')
        from sktime.classification.feature_based import Catch22Classifier
        clf = padder * Catch22Classifier()

    return clf


def classify(args):
    '''
    Loads data and trains classifier, and saves predictions to specified file.
    Classifier used is determined by command line arguments.

    args: parsed dictionary of command line arguments.
    '''
    print('Loading data...')
    data, labels = load_data(args['data'])

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, random_state=589, test_size=args['test_size'], shuffle=True
    )
    X_train, X_test = make_dataframe(X_train), make_dataframe(X_test)
    y_train = pd.Series(y_train)

    # maximum-length padding
    max_length = max(map(lambda sample: len(sample), data))

    # Create classifier pipeline based on selected method
    clf = get_classifier(args['method'], max_length)

    print('Starting training...')
    clf.fit(X_train, y_train)

    print('Making predictions...')
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    print(f'Accuracy: {acc}')

    # Save model, predictions, test data to file
    output_dir = args['output_dir']
    print(f'Saving model and predictions to {output_dir}...')
    os.makedirs(output_dir, exist_ok=True)  # create output directory
    with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as pred_file:
        d = {'X': X_test, 'pred': pred, 'gt': y_test}
        pickle.dump(d, pred_file)
    clf.save(os.path.join(output_dir, 'model'))  # saves output_dir/model.zip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Pickle file containing the data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save model and predictions to. Will create the directory if it doesn't exist.")
    parser.add_argument('--test_size', type=float, required=False, default=0.25,
                        help='Size of test split. Default is 0.25')
    parser.add_argument('--method', type=str, choices=CLASSIFIER_TYPES, required=False, default='rocket',
                        help='Which classifier to use. Default is rocket')
    args = vars(parser.parse_args())

    classify(args)
