#!/usr/bin/env python

import sys
import os
from operator import itemgetter
import matplotlib.pyplot as plt


def main(argv=[__name__]):

    if len(sys.argv) != 6:
        print "usage: <roc> <image> <graphic name> <x-axis name> <y-axis name>"
        sys.exit(0)

    rocfname = sys.argv[1]
    ofname = sys.argv[2]

    graphic_name = sys.argv[3]
    x_axis_name = sys.argv[4]
    y_axis_name = sys.argv[5]

    f, ext = os.path.splitext(ofname)
    if not IsSupportedImageType(ext):
        print("Format \"%s\" is not supported!" % ext)
        sys.exit(0)

    # read id of actives

    tpr, fpr = LoadROC(rocfname)
    print("Loaded %d from %s" % (len(tpr), rocfname))

    print("Plotting ROC Curve ...")
    color = "#008000"  # dark green
    DepictROCCurve(fpr, tpr, graphic_name, x_axis_name, y_axis_name, color, ofname)

def LoadROC(fname):
    sfile = open(fname, 'r')
    tpr = []
    fpr = []
    for line in sfile.readlines():
        tpr_value, fpr_value = line.strip().split()
        tpr.append(tpr_value)
        fpr.append(fpr_value)

    return tpr, fpr


def SetupROCCurvePlot(plt, graphic_name, x_axis_name, y_axis_name):

    plt.xlabel(x_axis_name, fontsize=14)
    plt.ylabel(y_axis_name, fontsize=14)
    plt.title(graphic_name, fontsize=14)


def SaveROCCurvePlot(plt, fname, randomline=True):

    #if randomline:
    #    x = [0.0, 1.0]
    #    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    #plt.xlim(0.0, 1.0)
    #plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(fname)


def AddROCCurve(plt, tpr, fpr, color, label):
    plt.plot(fpr, tpr, color=color, linewidth=2, label=label)


def DepictROCCurve(tpr, fpr, graphic_name, x_axis_name, y_axis_name, color, fname, randomline=True):

    plt.figure(figsize=(4, 4), dpi=280)

    SetupROCCurvePlot(plt, graphic_name, x_axis_name, y_axis_name)
    AddROCCurve(plt, tpr, fpr, color,graphic_name)
    SaveROCCurvePlot(plt, fname, randomline)


def IsSupportedImageType(ext):
    fig = plt.figure()
    return (ext[1:] in fig.canvas.get_supported_filetypes())


if __name__ == "__main__":
    main(sys.argv)