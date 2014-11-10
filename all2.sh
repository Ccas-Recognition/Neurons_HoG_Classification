#!/bin/bash
cd vs2013/PythonProj
python train_test_picker.py
cd ../..
train.sh
test.sh
cmp.sh