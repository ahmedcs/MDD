#!/usr/bin/env bash

NAME="synthetic_iid"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME