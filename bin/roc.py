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

parser.add_argument('--pred_csv_path', default='test/test.csv',
                    metavar='PRED_CSV_PATH', type=str,
                    help="Path to the prediction in csv")
parser.add_argument('--true_csv_path', default='dev.csv',
                    metavar='TRUE_CSV_PATH', type=str,
                    help="Path to the ground truth in csv")
parser.add_argument('--plot_path', default='test/', metavar='PLOT_PATH',
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
    transform_csv_en(args.pred_csv_path, args.plot_path + 'pred_csv_done.csv')
    transform_csv(args.true_csv_path, args.plot_path + 'true_csv_done.csv')

    images_pred, probs_pred, header_pred = read_csv(
        args.plot_path + 'pred_csv_done.csv')
    images_true, probs_true, header_true = read_csv(
        args.plot_path + 'true_csv_done.csv', True)

    # assert header_pred == header_true
    assert images_pred == images_true

    # num_labels = len(header_true) - 5
    num_labels = 1
    header = [header_true[14]]

    for i in range(num_labels):
        label = header[i]
        y_pred = probs_pred[:, i]
        y_true = probs_true[:, i]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(label, 'auc', auc)
        acc = metrics.accuracy_score(
            y_true, (y_pred >= args.prob_thred).astype(int), normalize=True
        )

        plt.figure(figsize=(8, 8), dpi=150)
        plt.xlim((0, 1.0))
        plt.ylim((0, 1.0))
        plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('{} ROC, AUC : {:.3f}, Acc : {:.3f}'.format(label, auc, acc))
        plt.plot(fpr, tpr, '-b')
        plt.grid()
        plt.savefig(
            os.path.join(args.plot_path, args.base_name
                         + '_' + label + '_roc.png'), bbox_inches='tight')


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
