#!/bin/bash

export UID=$(id -u)
export GID=$(id -g)

while [[ $# -gt 0 ]]; do
    case $1 in 
    --clean-container | --clean-all)
        docker rm ai-palatal-rugoscopy-env > /dev/null 2>&1
        ;;
    --clean-image | --clean-all)
        docker rmi ai-palatal-rugoscopy > /dev/null 2>&1
        ;;
    --data-path)
        data_path=$2
    esac
    shift
done

# Check wheter path to data folder is provided by the user, otherwise no data will be available inside the container.
if [[ -z $data_path ]]; then
    echo "ERROR: Data folder not specified"
    exit 1
fi

docker build \
    --build-arg USER=$USER \
    --build-arg PW=88452452 \
    --build-arg UID=$UID \
    --build-arg GID=$GID \
    --tag ai-palatal-rugoscopy .

docker run -it \
    --gpus all \
    --volume $(dirname `pwd`):/home/$USER/workspace/ia-palatal-rugoscopy \
    --volume $data_path:/home/$USER/data \
    --hostname ai-palatal-rugoscopy-env \
    --name ai-palatal-rugoscopy-env \
    ai-palatal-rugoscopy
