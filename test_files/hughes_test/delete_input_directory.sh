#!/bin/bash

mPATH="/media/tdh5188/easystore/data/convnet_input"

# rm all png in temp storage
rm -rf $mPATH/input_png/Classical/*
rm -rf $mPATH/input_png/Rock/*

# rm all jpgs in training set
rm -rf $mPATH/input/train/Classical/*
rm -rf $mPATH/input/train/Rock/*

# rm all jpgs in validation set
rm -rf $mPATH/input/valid/Classical/*
rm -rf $mPATH/input/valid/Rock/*


