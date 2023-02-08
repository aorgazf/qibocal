#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1

source activate qibocal-env
cd /home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/
python custom_rb.py