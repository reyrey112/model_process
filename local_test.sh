set -a
source $HOME/model_process/.env
set +a

docker compose -f "$HOME/model_process/local-compose.yaml" up --build -d --wait --wait-timeout 60 || {
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