#! /usr/bin/bash

conda create -n basicsr python=3.10
source ~/anaconda3/bin/activate basicsr
conda install pip
cd archs/basicsr
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop
cd ../../
pip install -r requirements.txt