#! /usr/bin/bash

conda create -n iseebetter python=3.10
source ~/anaconda3/bin/activate iseebetter
conda install pip
cd archs/iSeeBetter
pip install -r requirements.txt
cd pyflow
pip install Cython
pip install -e .
cd ../../../
pip install -r requirements.txt