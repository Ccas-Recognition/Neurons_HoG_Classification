from os import listdir, remove
from os.path import isfile, join, splitext, isfile
import numpy as np
import cv2

ftest = open("test.txt", "w")
ftrain = open("train.txt", "w")

def bmpToJpg(mypath, label):
    onlyfiles = np.array([ f for f in listdir(mypath) if isfile(join(mypath,f)) ])

    indx = np.random.choice(onlyfiles.size, 1000, replace = False)

    for file in onlyfiles[indx[:800]]:
        fileName, fileExtension = splitext(file)
        if(fileExtension != '.jpg'):
            continue
        ftrain.write('new/' + fileName + '.bmp ' + str(label) + '\n')
        # print(fileName)
        # img = cv2.imread(mypath + file)
        # cv2.imwrite('./new/' + fileName + '.bmp', img)

    for file in onlyfiles[indx[800:]]:
        fileName, fileExtension = splitext(file)
        if(fileExtension != '.jpg'):
            continue
        ftest.write('new/' + fileName + '.bmp ' + str(label) + '\n')
        # print(fileName)
        # img = cv2.imread(mypath + file)
        # cv2.imwrite('./new/' + fileName + '.bmp', img)

bmpToJpg('./bg/', 1)
bmpToJpg('./fg/', 2)