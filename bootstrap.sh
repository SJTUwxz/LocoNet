#!/usr/bin/env sh

export PYTHONPATH="`pwd`/PythonAPI/:`pwd`"
DATA_DIR=/data/users/$USER/002-retinanet 
mkdir -p $DATA_DIR
if [ ! -L data ]; then
    ln -s "$DATA_DIR" data
fi
