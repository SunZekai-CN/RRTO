# on GPU server
docker build -f Dockerfile.server -t offload .
docker run -v $PWD:/workspace -itd --gpus all --net host --ipc host --name offload offload
docker attach offload

# on robot
docker build -f Dockerfile.robot -t offload .
docker run -v $PWD:/workspace -itd --runtime nvidia --net host --ipc host --name offload offload
docker attach offload