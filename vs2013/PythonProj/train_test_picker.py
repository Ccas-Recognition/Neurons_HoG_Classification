from os import listdir, remove
from os.path import isfile, join, splitext, isfile
import numpy as np
import cv2

ftest = open("../../data/neuron/test.txt", "w")
ftrain = open("../../data/neuron/train.txt", "w")

if 0:
    sampling_count = 4915
    training_count = 4300
else:
    sampling_count = 1000
    training_count = 500

training_data_dir_txt = 'total_data/'
training_data_dir = '../../data/neuron/training_data/'
total_data_dir = '../../data/neuron/total_data/'

def bmpToJpg(subfolder, label):
    mypath = total_data_dir + subfolder
    onlyfiles = np.array([ f for f in listdir(mypath) if isfile(join(mypath,f)) and splitext(f)[1] == ".jpg"  ])

    indx = np.random.choice(onlyfiles.size, min( sampling_count, len( onlyfiles) ), replace = False)

    for file in onlyfiles[indx[:training_count]]:
        fileName, fileExtension = splitext(file)
        if(fileExtension != '.jpg'):
            continue
        ftrain.write(training_data_dir_txt + subfolder + file + ' ' + str(label) + '\n')
        #print([label, fileName])

    for file in onlyfiles[indx[training_count:]]:
        fileName, fileExtension = splitext(file)
        if(fileExtension != '.jpg'):
            continue
        ftest.write(training_data_dir_txt + subfolder + file + ' ' + str(label) + '\n')
        #print([label, fileName])


bmpToJpg( 'bg/', 1)
bmpToJpg( 'fg/', 2)

print('Sampling generated: %d training, %d testing'%((training_count*2), ((sampling_count - training_count)*2)))