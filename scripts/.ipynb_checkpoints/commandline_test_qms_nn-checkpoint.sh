#!/bin/bash

settingsfile="./test_qms_nn_settings.yml"

source activate jupyter

python -u ../code/test_qms_nn.py -model_yml $settingsfile 
