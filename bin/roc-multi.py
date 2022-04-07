import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# python bin/roc.py weights/dev_test.csv /data/ly/experiments/dev.csv weights/

parser = argparse.ArgumentParser(description='Plot ROC')

parser.add_argument('--plot_path0', default='test/', metavar='PLOT_PATH0',
                    type=str, help="Path to the ROC plots")
parser.add_argument('--plot_path1', default='test/', metavar='PLOT_PATH1',
                    type=str, help="Path to the ROC plots")
parser.add_argument('--plot_path2', default='test/', metavar='PLOT_PATH2',
                    type=str, help="Path to the ROC plots")
parser.add_argument('base_name', default=None, metavar='BASE_NAME',
                    type=str, help="Base name of the ROC plots")
parser.add_argument('--prob_thred', default=0.5, type=float,
                    help="Probability threshold")


def read_csv(csv_path, true_csv=False):
    image_paths = []
    probs = []
    dict_ = {'1.0': '1', '1': '1', '': '0', '0.0': '0', '0': '0', '-1.0': '0', '-1': '0', 'Unknown': '0'}

    with open(csv_path) as f:
        header = f.readline().strip('\n').split(',')
        for line in f:
            fields = line.strip('\n').split(',')
            image_paths.append(fields[0])

            if true_csv is False:
                probs.append(list(map(float, fields[1:])))
            else:
                prob = [int(dict_.get(fields[14]))]
                probs.append(prob)

    probs = np.array(probs)

    return (image_paths, probs, header)


def get_study(path):
    return path[0: path.rfind('/')]


def transform_csv(input_path, output_path):
    """
    to transform the first column of the original
     csv or test csv from Path to Study
    """
    infile = pd.read_csv(input_path)
    infile = infile.fillna('Unknown')
    infile.Path.str.split('/')
    infile['Study'] = infile.Path.apply(get_study)
    outfile = infile.drop('Path', axis=1).groupby('Study').max().reset_index()
    outfile.to_csv(output_path, index=False)


def transform_csv_en(input_path, output_path):
    """
    to transform the first column of the original
     csv or test csv from Path to Study
    """
    infile = pd.read_csv(input_path)
    infile = infile.fillna('Unknown')
    infile.Path.str.split('/')
    infile['Study'] = infile.Path.apply(get_study)
    outfile = infile.drop('Path', axis=1).groupby('Study').mean().reset_index()
    groups = infile.drop('Path', axis=1).groupby('Study')
    outfile['Pneumothorax'] = groups['Pneumothorax'].mean().reset_index()['Pneumothorax']
    outfile.to_csv(output_path, index=False)


def run(args):
    plt.figure(figsize=(8, 8), dpi=150)
    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')

    colors = ['red', 'green', 'blue']
    for n in range(3):
        index = 'plot_path' + str(n)
        images_pred, probs_pred, header_pred = read_csv(
            vars(args)[index] + 'pred_csv_done.csv')
        images_true, probs_true, header_true = read_csv(
            vars(args)[index] + 'true_csv_done.csv', True)

        assert images_pred == images_true

        header = header_true[14]

        y_pred = probs_pred[:, 0]
        y_true = probs_true[:, 0]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(header, 'auc', auc)
        acc = metrics.accuracy_score(
            y_true, (y_pred >= args.prob_thred).astype(int), normalize=True
        )

        plt.plot(fpr, tpr, c=colors[n], label='AUC : {:.3f}, Acc : {:.3f}'.format(auc, acc))

    plt.title('{} ROC'.format(header))
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(
        os.path.join('logdir-50-50', args.base_name
                     + '_' + header + '_roc.png'), bbox_inches='tight')


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
