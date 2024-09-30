# robotoffload
## Building
``` 
bash run_docker.sh
bash build_cricket.sh #execute in docker
```
## Run
``````
bash start_server.sh #on GPU server
bash start_client.sh #on client
`````` 
## Adapt to different versions
1. Adapt `cudnn` in `/cricket/cpu/cpu-*-cudnn.c`
    - `backendAttribute`
    - `attributeType >= CUDNN_TYPE_RNG_DISTRIBUTION`
    - `cudnnGetMaxDeviceVersion()` in `cudnn 8.6.0`
    - `typedef opaque rpc_cuda_device_prop[728];` in `/cricket/cpu/cpu_rpc_prot.x`
    - `exit(0)` in `deinit_rpc()`
2. `torch` enable `-cudart shared`
3. `torchvision` enable `-cudart shared`
