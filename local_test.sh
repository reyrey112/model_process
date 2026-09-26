set -a
source $HOME/model_process/.env
set +a

docker compose -f "$HOME/model_process/local-compose.yaml" up db -d --wait --wait-timeout 60 || {
    echo "Containers failed to become healthy"
    exit 1
}

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000