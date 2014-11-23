from os import listdir, remove
from os.path import isfile, join, splitext, isfile
import numpy as np
import cv2

ftest = open("../../data/neuron/test.txt", "w")
ftrain = open("../../data/neuron/train.txt", "w")

total_train = 0

if total_train:
    training_count = 4000
    training_bg_count = 10000
else:
    training_bg_count = 6000
    training_count = 200

training_data_dir_txt = 'total_data/'
total_data_dir = '../../data/neuron/total_data/'

def bmpToJpg(subfolder, sampling_count, label):
    mypath = total_data_dir + subfolder
    onlyfiles = np.array([ f for f in listdir(mypath) if isfile(join(mypath,f)) and splitext(f)[1] == ".jpg"  ])

    indx1 = np.random.choice(onlyfiles.size, min( sampling_count, len( onlyfiles) ), replace = False)
    indx2 = np.random.choice(onlyfiles.size, min( sampling_count, len( onlyfiles) ), replace = False)

    for file in onlyfiles[indx1]:
        fileName, fileExtension = splitext(file)
        ftrain.write(training_data_dir_txt + subfolder + file + ' ' + str(label) + '\n')
        #print([label, fileName])

    for file in onlyfiles[indx2]:
        fileName, fileExtension = splitext(file)
        ftest.write(training_data_dir_txt + subfolder + file + ' ' + str(label) + '\n')
        #print([label, fileName])

if 1:
    bmpToJpg( 'fg2/', training_count, 1)
    bmpToJpg( 'badfg2/', training_bg_count, 0)
    bmpToJpg( 'bg/', training_bg_count, 0)
else:
    bmpToJpg( 'fg/', training_count, 1)
    bmpToJpg( 'bg/', training_bg_count, 0)

print('Sampling generated. fg: %d, bg: %d'%(training_count,training_bg_count))