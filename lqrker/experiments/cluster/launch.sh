#!/bin/bash

echo "Running ./config/cluster/launch.sh ..."
echo "Total Arguments:             " $#
echo "(1) model:                   " $1
echo "(2) repetition number:       " $2

# export LD_LIBRARY_PATH=/lustre/home/amarcovalle/.mujoco/mujoco200/bin:/home/amarcovalle/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH

/home/amarcovalle/.anaconda3/envs/lqrker/bin/python run_experiments.py cluster.which_model=$1 cluster.rep_nr=$2


# condor_submit_bid 500 cluster/launch_GPFLOW.sub
# condor_q
# condor_rm
# alias check="condor_q amarcovalle"