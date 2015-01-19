from os import listdir, remove
from os.path import isfile, join, splitext, isfile
import numpy as np
import cv2

ftest = open("../../data/neuron/test.txt", "w")
ftrain = open("../../data/neuron/train.txt", "w")

total_train = 1

if total_train:
    training_count = 10000
    training_bg_count = 10000
    #training_count = 5000
    #training_bg_count = 9000
else:
    training_bg_count = 6000
    training_count = 200

training_data_dir_txt = 'total_data/'
total_data_dir = '../../data/neuron/total_data/'

def bmpToJpg(subfolder, sampling_count, label):
    mypath = total_data_dir + subfolder
    onlyfiles = np.array([ f for f in listdir(mypath) if isfile(join(mypath,f)) and splitext(f)[1] == ".jpg"  ])

    len_indx = min( sampling_count, len( onlyfiles) )
    mid_indx = len_indx/2
    indx1 = np.random.choice(onlyfiles.size, len_indx, replace = False)
    indx2 = np.random.choice(onlyfiles.size, len_indx, replace = False)

    for file in onlyfiles[indx1[:mid_indx]]:
        fileName, fileExtension = splitext(file)
        ftrain.write(training_data_dir_txt + subfolder + file + ' ' + str(label) + '\n')
        #print([label, fileName])

    for file in onlyfiles[indx1[mid_indx:]]:
        fileName, fileExtension = splitext(file)
        ftest.write(training_data_dir_txt + subfolder + file + ' ' + str(label) + '\n')
        #print([label, fileName])

if 0:
    bmpToJpg( 'fg2/', training_count, 1)
    bmpToJpg( 'badfg2/', training_bg_count, 0)
    bmpToJpg( 'bg/', training_bg_count, 0)
else:
    bmpToJpg( 'fg/', training_count, 1)
    bmpToJpg( 'bg/', training_bg_count, 0)

print('Sampling generated. fg: %d, bg: %d'%(training_count,training_bg_count))