#!/bin/sh

BASEDIR=$(pwd)
cd ..

python train.py raw_data/data $1

cd $BASEDIR
