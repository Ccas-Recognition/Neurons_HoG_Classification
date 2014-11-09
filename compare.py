#!/usr/bin/python

import sys
import os

def load_labels(filename):
    lines = open(filename, 'r').readlines()
    labels = {}
    data_path, data_filename = os.path.split(filename)
    for line in lines:
        parts = line.split(' ')
        img_path = parts[0]
        label = parts[1]
        norm_path = os.path.normpath(data_path + '/' + img_path)
        labels[norm_path] = label
    return labels


def test_labels(gt_labels, predicted_labels):
    if (len(gt_labels) != len(predicted_labels)):
        print "Error! Files with predicted and ground truth labels " \
              "have different number of samples."
        return
    if (len(gt_labels) == 0):
        print "Error! Dataset is empty."
        return

    shared_items = set(gt_labels.items()) & set(predicted_labels.items())
    correct_predictions = len(shared_items)
    precision = float(correct_predictions) / len(gt_labels)
    print "Precision: %f%%" % (100*precision)


def test_labels2(gt_labels, predicted_labels):
    if (len(gt_labels) != len(predicted_labels)):
        print "Error! Files with predicted and ground truth labels " \
              "have different number of samples."
        return
    if (len(gt_labels) == 0):
        print "Error! Dataset is empty."
        return
    total_count = len(gt_labels)
    first_kind = 0
    second_kind = 0

    for item in gt_labels.items():
        if( item[1] != predicted_labels[ item[0] ] ):
            if(item[1] == '1\n'):
                first_kind += 1
            else:
                second_kind += 1
    precision1 = 100*float(first_kind)/total_count
    precision2 = 100*float(second_kind)/total_count

    print "Error 1 (1 - right, 2 - predicted): %f%%" % (precision1)
    print "Error 2 (2 - right, 1 - predicted): %f%%" % (precision2)

if len(sys.argv) != 3:
    print 'Usage: %s <ground_truth.txt> <program_output.txt>' % sys.argv[0]
    sys.exit(0)

gt_labels = load_labels(sys.argv[1])
predicted_labels = load_labels(sys.argv[2])

test_labels(gt_labels, predicted_labels)
test_labels2(gt_labels, predicted_labels)