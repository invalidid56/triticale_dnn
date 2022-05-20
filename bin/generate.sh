#!/bin/sh

BASEDIR=$(pwd)
cd ..

python generate.py $1 $2

cd $BASEDIR
