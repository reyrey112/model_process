bash server_env_setup.sh

set -a
source "$HOME/model_process/.env"
set +a

sudo apt update && sudo apt upgrade -y
sudo apt install gnome-terminal -y
# sudo snap install docker
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

# python3.13 pip install uv
set -a
source "$HOME/model_process/.env"
set +a

uv sync

EXIT_CODE=0

sudo docker compose pull
sudo docker container prune -f
sudo docker image prune --force
docker compose -f "$HOME/model_process/server-compose.yaml" down 
sudo docker compose -f "$HOME/model_process/server-compose.yaml" up -d --wait --wait-timeout 60 || {
    echo "Containers failed to become healthy"
    exit 1
}

uv run "$HOME/model_process/process_loop/sqlite_setup.py" &

uv run "$HOME/model_process/process_loop/process_start_up_script.py" || EXIT_CODE=$?

if [$EXIT_CODE -eq 1]; then
    exit $EXIT_CODE
fi

echo "Redis seeded, process starting"

trap 'kill 0' EXIT

uv run "$HOME/model_process/process_loop/process_dynamic.py" &
uv run "$HOME/model_process/process_loop/process_control.py" &
wait -n
echo "One of the scripts exited, shutting down..."