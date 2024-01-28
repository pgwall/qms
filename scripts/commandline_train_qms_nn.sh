#!/bin/bash

settingsfile="./train_qms_nn_settings.yml"

source activate jupyter

python -u ../code/train_qms_nn.py -model_yml $settingsfile 