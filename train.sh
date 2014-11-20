#!/bin/bash
cd build/bin
./task2 -d ../../data/neuron/train.txt -m ../../model_binary.txt -c ../../context.ini --train
cd ../..