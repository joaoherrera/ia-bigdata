#!/bin/bash

export UID=$(id -u)
export GID=$(id -g)

for arg in $@; do
    if [[ $arg == clean ]]; then
        docker rm cv-hardware-parts
	docker rmi ia-bigdata
    fi;
done

docker build --build-arg USER=$USER --build-arg PW=cv123 --build-arg UID=$UID --build-arg GID=$GID -t ia-bigdata .
docker run -it --gpus all -v $(dirname `pwd`):/home/$USER/workspace/cv-hardware-parts -v $HOME/SMACS-0723:/home/$USER/server --name cv-hardware-parts ia-bigdata
