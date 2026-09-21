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

# export venv = "venv_model_process"

# if [ ! -d "${venv}" ]; then
#     python3.13 -m "${venv}"
# fi

# source "${venv}"/bin/activate
# "${venv}"/bin/pip install uv

sudo apt update && sudo apt install -y --no-install-recommends 'build-essential'
# "${venv}"/bin/uv pip install -r requirements.txt

python3.13 pip install uv

uv sync

EXIT_CODE=0

sudo docker compose pull
sudo docker container prune -f
sudo docker image prune --force
sudo docker compose -f docker-compose.yaml up -d --wait --wait-timeout 60 || {
    echo "Containers failed to become healthy"
    exit 1
}

uv run "$HOME/model_process/process_loop/sqlite_setup.py" &

uv run "$HOME/model_process/process_loop/process_start_up_script.py" || EXIT_CODE=$?

if [$EXIT_CODE -eq 1]; then
    exit $EXIT_CODE
fi

echo "Redis seeded, process starting"

uv run "$HOME/model_process/process_dyanmic.py" &
uv run "$HOME/model_process/process_control.py" &
wait -n
