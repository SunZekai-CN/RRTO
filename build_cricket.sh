#!/bin/bash
cd ./cricket
make clean
make -j 32 libtirpc
make -j 32 cuda-gdb
# cuda 12
# make -j 1 LOG=DEBUG 
# make -j 1 tests LOG=DEBUG
#cuda 11
make -j 1 LOG=INFO NOSAMPLES=yes
echo "build cricket finish!!!"
