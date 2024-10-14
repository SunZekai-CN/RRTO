# RRTO: A High-Performance Transparent Offloading System for Model Inference on Robotic IoT


RRTO is a high-performance transparent offloading system specifically designed for ML model inference on robotic IoT with a novel record/replay mechanism.

We are deeply grateful to [Cricket](https://github.com/RWTH-ACS/cricket) for their remarkable open-source project, which served as the foundation for RRTO's development. 
RRTO seamlessly integrates its innovative recorder and replayer into Cricket's corresponding RPC functions, leveraging the same Remote Procedure Call (RPC) for efficient communication as Cricket.

At present, due to the constraints of our hardware platform (detailed in `Dockerfile.robot`), RRTO exclusively supports `cuda=11.4` and `cudnn=8.4`. 
To harness RRTO's capabilities with PyTorch and torchvision, you'll need to recompile the compatible versions: `PyTorch 1.12.0` and `torchvision 0.13.0`, both tailored for `cuda=11.4` and `cudnn=8.4`. 
Rest assured, we are diligently working on expanding support for a broader range of cuda, PyTorch, and torchvision versions in the near future.


## Installation
1. Clone this repo and enter the project folder.

2. Building and initiating the corresponding docker containers on both the server and robot sides based on the dockerfile file we provided (`Dockerfile.robot` and `Dockerfile.server`), or execute the script we provided directly.

```
bash run_docker.sh
```
Note that `CmakeLists.txt`, `common.mk`, `Makefile` need to be replaced in the corresponding pytorch and torchvison source code (`PyTorch 1.12.0` and `torchvision 0.13.0`) according to the guide of `prepare_pytorch.sh`, which has already been done in our dockerfile.
Due to the need to recompile PyTorch and torchvision from source code, the entire building process usually takes several hours.


3. Install RRTO in the docker containers on both the server and robot sides:
```
bash build_cricket.sh #execute in docker
```
To accommodate our robotic hardware, we have implemented a series of key modifications to Cricket's existing infrastructure.

```
1. Adapt `cudnn` in `/cricket/cpu/cpu-*-cudnn.c`
    - `backendAttribute`
    - `attributeType >= CUDNN_TYPE_RNG_DISTRIBUTION`
    - `cudnnGetMaxDeviceVersion()` in `cudnn 8.6.0`
    - `typedef opaque rpc_cuda_device_prop[728];` in `/cricket/cpu/cpu_rpc_prot.x`
    - `exit(0)` in `deinit_rpc()`
2. `torch` enable `-cudart shared`
3. `torchvision` enable `-cudart shared`
```


## How to Use?
1. start RRTO server on the GPU server via the following script.
```
bash start_server.sh #on GPU server
```
2. Start the python script normally for model inference via RRTO client as the following script.
```
bash start_client.sh #on client
``` 
Note that `REMOTE_GPU_ADDRESS` is the IP address of the GPU server.

## Cite Us
Upcoming, the paper is under review.