#!/bin/bash

#$-l rt_F=1
#$-l h_rt=5:50:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source ../loadpython.sh
source ../venv/bin/activate

python train.py --exp_name=exp03

