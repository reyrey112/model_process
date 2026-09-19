#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install gnome-terminal
sudo snap install docker
sudo apt install git -y

if [ -d "/home/ubuntu/model_process" ]; then
    cd /home/ubuntu/model_process && git pull
else
    git clone https://github.com/reyrey112/model_process /home/ubuntu/model_process
    cd /home/ubuntu/model_process
fi


sudo apt install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev
sudo apt install -y certbot python3-certbot-nginx

export venv = "venv_model_process"

if [ ! -d "${venv}" ]; then
    python3.13 -m "${venv}"
fi

source "${venv}"/bin/activate
"${venv}"/bin/pip install uv

sudo apt update && sudo apt install -y --no-install-recommends 'build-essential'
"${venv}"/bin/uv sync

mkdir -p ~/.aws
if [ ! -f ~/.aws/config ]; then
    echo "[default]
region = us-east-2" > ~/.aws/config
fi

venv_model_process/bin/python download_from_ssm.py

export SSL_CERTIFICATE=/etc/letsencrypt/live/${SERVER_NAME}/fullchain.pem
export SSL_CERTIFICATE_KEY=/etc/letsencrypt/live/${SERVER_NAME}/privkey.pem

sudo snap install aws-cli --classic
sudo aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com

# sudo docker-compose pull
sudo docker container prune -f
sudo docker image prune --force
sudo docker-compose -f docker-compose.yaml up -d