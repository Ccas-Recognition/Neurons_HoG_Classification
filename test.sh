#!/bin/bash
cd build/bin
./task2 -d ../../data/neuron/test.txt -m ../../model_binary.txt -l predictions.txt --predict
cd ../..