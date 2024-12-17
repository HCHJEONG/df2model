#!/bin/bash

# make shell variables with script arguments
# echo -n "input Asymmetric Key Passphrase:"
# stty -echo
# read PW
# stty echo

# echo ""

ML_BACKEND_IMAGE=ml_backend
IMAGE_TIMESTAMP=$(date +"%G%m%d%H%M")
REMOTE_SERVER=ubuntu@3.35.49.93

# cloning lawvote repository
# ML_BACKEND_DIR=~/downloads/git_repo/df2model

# if [ -d ${ML_BACKEND_DIR} ]; then
#         echo "ml_backend repository already exist . ."
#         echo "removing ml_backend repository . ."
#         rm -rdf "${ML_BACKEND_DIR}"
#         echo "ml_backend repository removed . ."

# else echo "ml_backend repository not exist . ."
# fi

# cd ~/downloads/git_repo
# expect -c "
# spawn git clone git@github.com:HCHJEONG/df2model
# expect \"Enter passphrase for key\"
# send \"${PW}\r\"
# expect \"$\"
# exit 0
# "
# echo "df2model repository cloned . . ."

# cp env variables to ml_backend repository
cd ./df2model-aws-backup
cp .env toshiba_nginx_ca.crt ../../
cd ../../

# builing docker image with shell variables
echo "building docker image: ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP} started . . ."
(docker build --tag ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP} .) || (exit)
echo "building docker image: ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP} ended . . ."

# converting docker image to tar file exe
echo "converting docker image: ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP} to ml_backend${IMAGE_TIMESTAMP}.tar started . . ."
docker save ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP} > ml_backend${IMAGE_TIMESTAMP}.tar
echo "converting docker image: ${ML_BACKEND_IMAGE}:${IMAGE_TIMESTAMP} converted to ml_backend${IMAGE_TIMESTAMP}.tar . . ."

# transferring docker image tar file with scp command
echo "transferring docker image tar file: ml_backend${IMAGE_TIMESTAMP}.tar started . . ."
expect -c "
set timeout 2000
spawn scp -i /home/hchjeong/.ssh/penvotkeypair1.pem ml_backend${IMAGE_TIMESTAMP}.tar ${REMOTE_SERVER}:/home/ubuntu
expect \"Enter passphrase for key\"
send \"${PW}\r\"
expect \"$\"
exit 0
"
echo "transferring docker image tar file: ml_backend${IMAGE_TIMESTAMP}.tar ended . . ."

# execute ml_backend_auto_deploy2.sh in remote server
# cd ./.fordeploy/

# IMAGE_FILE_PATH=~/docker_images/ml_backend_images/ml_backend${IMAGE_TIMESTAMP}.tar

# sshpass -v -P phrase -p ${PW} ssh -p 3330 ${REMOTE_SERVER} 'bash -s' ${PW} ${ML_BACKEND_IMAGE} ${IMAGE_TIMESTAMP} < ./ml_backend_auto_deploy2.sh

# cleaning
# cd ../
rm ./.env
rm ./toshiba_nginx_ca.crt
rm ./*.tar
