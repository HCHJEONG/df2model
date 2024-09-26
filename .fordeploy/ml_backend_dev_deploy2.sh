#!/bin/bash
PW=$1
ML_BACKEND_IMAGE=$2
IMAGE_TIMESTAMP=$3
IMAGE_FILE_PATH=~/docker_images/ml_backend_images/ml_backend${IMAGE_TIMESTAMP}.tar
echo ${ML_BACKEND_IMAGE}
echo ${IMAGE_FILE_PATH}
echo ${IMAGE_TIMASTAMP}


# check docker image file exist then load docker image file

echo "ssh login successed"
if [ -e "${IMAGE_FILE_PATH}" ]; then
        echo "loading docker image: ${IMAGE_FILE_PATH} started . . . "
        docker load -i "${IMAGE_FILE_PATH}"
else
        echo "${IMAGE_FILE_PATH}"
        pwd
        ls -lia
fi



# if "docker container ls"'s output has "nextjs" stop that container
ML_BACKEND_COUNT=$(docker container ls | grep -c 'ml_backend')
if [ ${ML_BACKEND_COUNT} -ne 0 ]; then
        CONTAINER_NAME=$(docker container ls | awk '{print NR,$NF}' | grep 'ml_backend' | cut -c 3-)
        echo ${CONTAINER_NAME}
        docker container stop ${CONTAINER_NAME}
        echo "docker container ${CONTAINER_NAME} stopped . . ."
fi

# generate docker container then execute pm2 with nextjs server


docker run -dit -p 5001:5001 -v /home/lawvot/volumes-lvot/ml-data/model:/app/model --name ${ML_BACKEND_IMAGE}-${IMAGE_TIMESTAMP} ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP}
