#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install -y gnome-terminal
# sudo snap install -y docker
sudo apt install git -y

export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/model_process_deploy -o StrictHostKeyChecking=accept-new"
REPO="git@github.com:reyrey112/model_process.git"

if [ -d "$HOME/model_process" ]; then
    echo "Repo exists"
    cd "$HOME/model_process" && git pull
else
    echo "Repo does not exist, pulling"
    git clone "$REPO" "$HOME/model_process"
    cd "$HOME/model_process"
fi


sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev

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
