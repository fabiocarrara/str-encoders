#!/bin/bash

# set -e

DATASETS=(
    "glove-25-angular"
)


COMMON="--exp-root runs/ --search-batch-size 625 --index-batch-size 10000"

##
# Deep Permutation
##
declare -A DP_Ls
DP_Ls=(
    ["glove-25-angular"]="1 2 3 4 5 8 10 12 15 20 25"
)

for DATASET in $DATASETS; do
    for L in ${DP_Ls[$DATASET]}; do
        python run.py $COMMON $DATASET deep-perm --permutation-length $L --rectify-negatives
    done
done

##
# Threshold Scalar Quantization
##
declare -A THRSQ_Qs THRSQ_Ss
THRSQ_Qs=(
    ["glove-25-angular"]="99 98 97 96 95 90 85 80 75 70 60 50 40 30 20 10 1"
)

THRSQ_Ss=(
    ["glove-25-angular"]="100000"  # "1000 100000"
)

for DATASET in $DATASETS; do
    for S in ${THRSQ_Ss[$DATASET]}; do
    for Q in ${THRSQ_Qs[$DATASET]}; do
        python run.py $COMMON $DATASET thr-sq --threshold-percentile $Q --sq-factor $S --rectify-negatives --l2-normalize
    done
    done
done

##
# SPQR
##
declare -A SPQR_Cs SPQR_Ms SPQR_Fs

SPQR_Cs=(
    ["glove-25-angular"]="128 256 512 1024 2048 4096 8192"
)

SPQR_Ms=(
    ["glove-25-angular"]="1 5 25"
)

SPQR_Fs=(
    ["glove-25-angular"]="256 1024 4096 16384 65536"
)


for DATASET in $DATASETS; do
    for F in ${SPQR_Fs[$DATASET]}; do
    for M in ${SPQR_Ms[$DATASET]}; do
    for C in ${SPQR_Cs[$DATASET]}; do
        python run_spqr.py $COMMON $DATASET -c $C -m $M -f $F
    done
    done
    done
done