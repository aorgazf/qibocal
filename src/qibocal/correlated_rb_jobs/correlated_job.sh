#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1

source activate qibocal-env
cd /home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/correlated_rb_jobs/
python create_df.py