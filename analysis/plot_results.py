import os
import pickle
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from adjustText import adjust_text

# Add classification folder to path to import functions from scripts
classification_path = os.path.join(os.path.dirname(__file__), '../classification')
sys.path.append(classification_path)
from analyze_results import modified_accuracy_score
from make_predictions import max_padded_predictions

DPI = 200
CHOICES = ['estimators', 'transforms', 'kernels', 'features',
           'max_padding', 'adaptive_padding', 'pad_comparison', 'timing'
           ]
FIGS_DIR = os.path.join(os.path.dirname(__file__), '../figs')

from plot_data import get_color


def plot_estimator_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different estimator types.
    '''
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

    # Get list of labels
    with open('collection_scripts/top-1k-curated', 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    classifiers = ['ridge', 'logisticreg', 'SVM']

    df = pd.DataFrame([], columns=['estimator', 'accuracy', 'f1-score', 'modified_accuracy'])

    for classifier in classifiers:
        pred_file = f'classification/minirocket_full_{classifier}/predictions.pkl'
        with open(pred_file, 'rb') as fh:
            d = pickle.load(fh)
            y_pred = d['pred']
            y_true = d['gt']

        df.loc[len(df.index)] = [
            classifier,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            modified_accuracy_score(y_true, y_pred, labels)
        ]

    # reshape dataframe
    df = pd.melt(df, id_vars='estimator', var_name='metric', value_name='value')

    # Plot bar graph for each estimator with each metric
    with sns.color_palette('deep', n_colors=3):
        ax = sns.barplot(df, x='estimator', y='value', hue='metric', alpha=0.8)

    # Add value to top of each bar
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=8)

    # Labels and stuff
    ax.set_xlabel('Estimator')
    ax.set_ylabel('Value')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    plt.savefig(os.path.join(FIGS_DIR, 'compare_estimators.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_transform_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different transforms.
    '''
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

    # Get list of labels
    with open('collection_scripts/top-1k-curated', 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    classifiers = {
        'rocket': 'classification/rocket_full_default/predictions.pkl',
        'catch-22': 'classification/predictions/predictions_200-class_100-samples_catch22.pkl',
        'minirocket': 'classification/minirocket_full_ridge/predictions.pkl'
    }

    df = pd.DataFrame([], columns=['classifier', 'accuracy', 'f1-score', 'modified_accuracy'])

    for classifier, pred_file in classifiers.items():
        with open(pred_file, 'rb') as fh:
            d = pickle.load(fh)
            y_pred = d['pred']
            y_true = d['gt']

        df.loc[len(df.index)] = [
            classifier,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            modified_accuracy_score(y_true, y_pred, labels)
        ]

    # reshape dataframe
    df = pd.melt(df, id_vars='classifier', var_name='metric', value_name='value')

    # Create bar graph for each classifier with each metric
    with sns.color_palette('deep', n_colors=3):
        ax = sns.barplot(df, x='classifier', y='value', hue='metric', alpha=0.8)

    # Add value to top of each bar
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=8)

    # Labels and stuff
    ax.set_xlabel('Transform')
    ax.set_ylabel('Value')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    plt.savefig(os.path.join(FIGS_DIR, 'compare_transforms.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_kernel_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different number of convolutional kernels with ROCKET.
    '''
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

    # Get list of labels
    with open('collection_scripts/top-1k-curated', 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    kernels = {
        10000: 'classification/rocket_full_default',
        5000: 'classification/rocket_full_5000-kernels',
        8000: 'classification/rocket_full_8000-kernels',
        11000: 'classification/rocket_full_11000-kernels'
    }

    df = pd.DataFrame([], columns=['kernel_size', 'accuracy', 'f1-score', 'modified_accuracy'])

    for kernel_size in kernels.keys():
        pred_file = f'{kernels[kernel_size]}/predictions.pkl'
        with open(pred_file, 'rb') as fh:
            d = pickle.load(fh)
            y_pred = d['pred']
            y_true = d['gt']

        df.loc[len(df.index)] = [
            kernel_size,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            modified_accuracy_score(y_true, y_pred, labels)
        ]

    # reshape dataframe
    df = pd.melt(df, id_vars='kernel_size', var_name='metric', value_name='value')

    # plot
    with sns.color_palette('deep', n_colors=3):
        ax = sns.lineplot(df, x='kernel_size', y='value', hue='metric', marker='o')

    # Labels and stuff
    ax.set_xlabel('Kernel size')
    ax.set_ylabel('Value')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    plt.savefig(os.path.join(FIGS_DIR, 'compare_kernels.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_feature_comparison_by_direction():
    '''
    Creates a bar chart comparing accuracy, across different features
    '''
    # features/direction/accuracy
    data = {
        'time, size': {
            'both': 0.962,
            'outgoing': 0.942,
            'incoming': 0.960,
            'none': 0.971,
        },
        'time': {
            'both': 0.910,
            'outgoing': 0.899,
            'incoming': 0.873,
            'none': 0.919,
        },
        'size': {
            'both': 0.945,
            'outgoing': 0.887,
            'incoming': 0.937,
            'none': 0.960,
        },
        'direction': {
            'both': 0.707,
            'outgoing': 0.136,
            'incoming': 0.114,
        },
    }
    data_lists = []
    for feature, directions in data.items():
        for direction, accuracy in directions.items():
            data_lists.append([
                direction,
                feature,
                accuracy
            ])

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

    df = pd.DataFrame(data_lists, columns=['direction', 'feature', 'accuracy', ])

    # Create bar graph for each feature with each direction
    with sns.color_palette('deep', n_colors=4):
        ax = sns.barplot(df, x='direction', y='accuracy', hue='feature', alpha=0.8)

    # Add value to top of each bar
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=6)

    # Labels and stuff
    ax.set_xlabel('Direction')
    ax.set_ylabel('Accuracy')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=4, title=None, frameon=False)
    plt.savefig(os.path.join(FIGS_DIR, 'compare_features_by_direction.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_max_padding():
    '''
    Creates a chart showing the accuracy for different sizes of max padding.
    '''
    df = pd.read_csv('classification/max_pad.csv')

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    palette = sns.color_palette('deep', n_colors=2)

    # Plot modified accuracy vs pad size on one graph
    sns.lineplot(df, x='size', y='modified_acc', ax=ax1, marker='o', color=palette[0])

    # Plot overhead vs pad size on other graph
    sns.lineplot(df, x='size', y='overhead', ax=ax2, marker='o', color=palette[1])

    handles = [Line2D([], [], marker='o', color=palette[0], label='Modified Accuracy'),
               Line2D([], [], marker='o', color=palette[1], label='Overhead')]

    # Labels and stuff
    ax1.set_xlabel('Pad size (bytes)')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Overhead (%)')

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax1.set_ylim(-0.02, 1.02)
    ax2.set_ylim(-0.04, 2.04)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    ax2.set_yticks([tick * 2 for tick in ticks])
    ax2.set_yticklabels([f'{tick * 200:.0f}' for tick in ticks])

    ax1.legend(handles=handles)
    sns.move_legend(ax1, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False)

    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'max_padding.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_adaptive_padding():
    '''
    Creates a chart showing the accuracy for different sizes of adaptive padding.
    '''
    df = pd.read_csv('classification/adaptive_pad.csv')

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    palette = sns.color_palette('deep', n_colors=2)

    # Plot modified accuracy vs pad size on one graph
    sns.lineplot(df, x='multiple', y='modified_accuracy', ax=ax1, marker='o', color=palette[0])

    # Plot overhead vs pad size on other graph
    sns.lineplot(df, x='multiple', y='overhead', ax=ax2, marker='o', color=palette[1])

    handles = [Line2D([], [], marker='o', color=palette[0], label='Modified Accuracy'),
               Line2D([], [], marker='o', color=palette[1], label='Overhead')]

    # Labels and stuff
    ax1.set_xlabel('Padding multiple (bytes)')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Overhead (%)')

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax1.set_ylim(-0.02, 1.02)
    ax2.set_ylim(-0.04, 2.04)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    ax2.set_yticks([tick * 2 for tick in ticks])
    ax2.set_yticklabels([f'{tick * 200:.0f}' for tick in ticks])

    ax1.legend(handles=handles)
    sns.move_legend(ax1, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False)

    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'adaptive_padding.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_pad_scatter():
    '''
    Creates a scatter plot of all the different padding strategies plotting as overhead
    vs accuracy.
    '''

    # Load data
    adaptive_df = pd.read_csv('classification/adaptive_pad.csv')
    max_df = pd.read_csv('classification/max_pad.csv')
    directional_df = pd.read_csv('classification/directional_pad.csv')

    # Rename columns to make it easier to join
    adaptive_df.rename(columns={'multiple': 'pad'}, inplace=True)
    max_df.rename(columns={'size': 'pad'}, inplace=True)
    directional_df.rename(columns={'length': 'pad'}, inplace=True)

    # Add type column to differentiate once combined
    adaptive_df['type'] = 'block'
    max_df['type'] = 'max'
    directional_df['type'] = 'directional (query, response)'

    # Merge with inner join to drop unneeded columns
    merged_df = pd.concat([max_df, adaptive_df, directional_df])
    merged_df = merged_df[['pad', 'modified_accuracy', 'overhead', 'type']]
    merged_df = merged_df.reset_index(drop=True)

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)
    palette = sns.color_palette('deep', n_colors=2)

    # Plot overhead vs modified accuracy
    ax = sns.scatterplot(merged_df, x='overhead', y='modified_accuracy', hue='type', style='type')

    # Add labels to points and adjust so they don't overlap
    texts = []
    for index in merged_df.index:
        label = merged_df["pad"][index]
        texts.append(ax.text(
            merged_df['overhead'][index], merged_df['modified_accuracy'][index], label, fontsize=6
        ))
    adjust_text(texts)

    # Labels and stuff
    ax.set_xlabel('Overhead')
    ax.set_ylabel('Modified Accuracy')
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xticklabels([f'{i}%' for i in [0, 50, 100, 150, 200, 250]])

    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'compare_padding_scatter.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot_inter_arrival_timing_experiments():
    '''
    Creates a chart showing the accuracy for different delays and different classifiers,
    using inter-arrival times.
    '''
    results = {'minirocket_logreg': {0.001: (0.9809985850010107,
                                             0.005813797957205733),
                                     0.002: (0.9801900141499899,
                                             0.01163254912067527),
                                     0.005: (0.977360016171417,
                                             0.029067545189505335),
                                     0.01: (0.9747321609055993,
                                            0.0582239104902823),
                                     0.02: (0.9690721649484536,
                                            0.11647031055941481),
                                     0.05: (0.8960986456438246,
                                            0.29118420384986077),
                                     0.1: (0.615726703052355, 0.5815019385913974),
                                     0.2: (0.39336971902162926,
                                           1.164152119013465),
                                     0.5: (0.10713563776025874,
                                           2.905587409118061),
                                     1: (0.05134424903982211, 5.831649081899804)},
               'minirocket_ridge': {0.001: (0.9803921568627451,
                                            0.005817870896784802),
                                    0.002: (0.9801900141499899,
                                            0.011641915020025581),
                                    0.005: (0.979381443298969,
                                            0.029084436937573042),
                                    0.01: (0.9777643015969274,
                                           0.05818997246555567),
                                    0.02: (0.9759450171821306,
                                           0.1163810281831414),
                                    0.05: (0.9593693147362038,
                                           0.2908634263795022),
                                    0.1: (0.9241964827167981, 0.5820073333732103),
                                    0.2: (0.8122094198504144, 1.162904889062114),
                                    0.5: (0.4633110976349303, 2.907873630694599),
                                    1: (0.15868202951283605, 5.8162823401888195)},
               'minirocket_ridgecv': {0.001: (0.9765514453203962, 0.005833190595642735),
                                      0.002: (0.9751364463311097, 0.01163769728382304),
                                      0.005: (0.9749343036183545, 0.029110271933818262),
                                      0.01: (0.9745300181928441, 0.05808034921531257),
                                      0.02: (0.9735193046290681, 0.11634181133271472),
                                      0.05: (0.9619971700020215, 0.2913091049789797),
                                      0.1: (0.9114614918132201, 0.5819290117475253),
                                      0.2: (0.7531837477258945, 1.1643259203077596),
                                      0.5: (0.40327471194663433, 2.912192424253857),
                                      1: (0.1170406306852638, 5.812480068120125)}}
    names = {
        'minirocket_logreg': 'logreg',
        'minirocket_ridge': 'ridge',
        'minirocket_ridgecv': 'ridgecv',
    }
    rows = []
    columns = ['delay', 'overhead'] + [m for m in results.keys()]
    delay_overhead = {k: v[1] for k,v in results['minirocket_logreg'].items()}
    for d, o in delay_overhead.items():
        row = [d, o*100]
        for m, data in results.items():
            row.append(data[d][0])
        rows.append(row)
    df = pd.DataFrame(rows,columns=columns)

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)
    fig, ax_accuracy = plt.subplots()
    ax_overhead = ax_accuracy.twinx()

    palette = sns.color_palette('deep', n_colors=len(results)+1)
    handles = []
    # Plot modified accuracy vs delay on one graph
    for idx, m in enumerate(results.keys()):
        sns.lineplot(df, x='delay', y=m, ax=ax_accuracy, marker='o', color=palette[idx])
        handles.append(Line2D([], [], marker='o', color=palette[idx], label=names[m]))
    # Plot overhead vs delay on other graph
    sns.lineplot(df, x='delay', y='overhead', ax=ax_overhead, marker='o', color=palette[len(results)])
    handles.append(Line2D([], [], marker='o', color=palette[len(results)], label='Overhead'))

    # Labels and stuff
    ax_accuracy.set_xlabel('Packet Delay (s)')
    ax_accuracy.set_ylabel('Modified Accuracy')
    ax_overhead.set_ylabel('Overhead (%)')

    acc_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax_accuracy.set_ylim(-0.02, 1.02)
    ax_accuracy.set_yticks(acc_ticks)
    ax_accuracy.set_yticklabels(acc_ticks)

    over_ticks = [tick * 600 for tick in acc_ticks]
    ax_overhead.set_ylim(-0.02*600, 1.02*600)
    ax_overhead.set_yticks(over_ticks)
    ax_overhead.set_yticklabels([f'{tick:.0f}' for tick in over_ticks])

    ax_accuracy.legend(handles=handles)
    sns.move_legend(ax_accuracy, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=4, title=None, frameon=False)

    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'timing_results.png'), bbox_inches='tight', dpi=DPI)
    plt.show()


def plot(plot_type):
    if plot_type == 'transforms':
        print('Plotting transforms...')
        plot_transform_comparison()
    elif plot_type == 'estimators':
        print('Plotting estimators...')
        plot_estimator_comparison()
    elif plot_type == 'features':
        print('Plotting features...')
        plot_feature_comparison_by_direction()
    elif plot_type == 'max_padding':
        print('Plotting max padding...')
        plot_max_padding()
    elif plot_type == 'adaptive_padding':
        print('Plotting adaptive padding...')
        plot_adaptive_padding()
    elif plot_type == 'pad_comparison':
        print('Plotting comparison of all padding methods...')
        plot_pad_scatter()
    elif plot_type == 'timing':
        print('Plotting timing results...')
        plot_inter_arrival_timing_experiments()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('plot_type', choices=CHOICES, nargs='+', type=str,
                        help='List of plots to generate. Can specify multiple options. See help for list of \
              possible plots.')
    args = parser.parse_args()

    for plot_type in args.plot_type:
        plot(plot_type)
