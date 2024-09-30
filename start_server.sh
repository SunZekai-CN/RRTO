#!/bin/bash
service rpcbind start
LD_PRELOAD='./cricket/bin/libtirpc.so.3' ./cricket/bin/cricket-rpc-server
