#!/bin/bash
# REMOTE_GPU_ADDRESS=202.45.128.184 LD_PRELOAD=./cricket/bin/cricket-client.so  ./cricket/tests/test_apps/cricket.testapp
# LD_LIBRARY_PATH=./cricket/cpu REMOTE_GPU_ADDRESS=192.168.50.11 LD_PRELOAD='./cricket/bin/cricket-client.so ./cricket/bin/libtirpc.so.3' python3 ./cricket/tests/test_apps/pytorch_minimal.py
LD_LIBRARY_PATH=./cricket/cpu REMOTE_GPU_ADDRESS=192.168.50.11 LD_PRELOAD='./cricket/bin/cricket-client.so ./cricket/bin/libtirpc.so.3' python3 ./test/main.py
