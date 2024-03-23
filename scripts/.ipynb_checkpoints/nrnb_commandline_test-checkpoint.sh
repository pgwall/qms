#!/bin/bash

#SBATCH --job-name test
#SBATCH --output=/cellar/users/pgwall/nrnb_logs/test_submission.%j
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:v100:1

#SBATCH -N 1      


/cellar/users/pgwall/qms/scripts/commandline_test_qms_nn.sh
