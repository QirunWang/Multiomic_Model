#!/bin/bash
#DSUB -n model_pretrain
#DSUB -N 4
#DSUB -A root.project.P24Z28400N0259_tmp
#DSUB -R "cpu=20;gpu=4;mem=100000"
#DSUB -oo /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/logs/test_task.%J.out\
#DSUB -eo /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/logs/test_task.%J.err\

## Set scripts
RANK_SCRIPT="./tmp/test_20250410_DDP/train.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/"

## Set NNODES
NNODES=3

## Create nodefile

JOB_ID=${BATCH_JOB_ID}
NODEFILE=/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/${JOB_ID}.nodefile
touch $NODEFILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE}
