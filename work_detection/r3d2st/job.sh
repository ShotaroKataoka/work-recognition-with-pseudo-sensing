#!/bin/bash

#$-l rt_F=1
#$-l h_rt=24:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source ../loadpython.sh
source ../venv/bin/activate

python train.py

