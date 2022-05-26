#!/bin/sh

BASEDIR=$(pwd)
cd ..

python train.py data $1

cd $BASEDIR