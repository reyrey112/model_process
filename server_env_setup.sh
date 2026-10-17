#!/usr/bin/env bash
#
# setup_env.sh
# Ensures a .env file exists, checks for required keys, and prompts
# the user to enter any that are missing.
#
# Usage: ./setup_env.sh

set -euo pipefail

# --- Configuration ----------------------------------------------------------

# Resolve .env relative to this script, not the current working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Keys that must be present
REQUIRED_KEYS=(DATABRICKS_HOST DATABRICKS_TOKEN)

# Keys whose input should be visible while typing (everything else is hidden)
VISIBLE_KEYS=()

# --- Helpers ----------------------------------------------------------------

# Create .env with restrictive permissions if it doesn't exist
ensure_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        (umask 077; touch "$ENV_FILE")
        echo "Created $ENV_FILE"
    fi
}

# True if the key is defined in the file (handles 'export' and whitespace)
has_key() {
    local key="$1"
    grep -qE "^\s*(export\s+)?${key}\s*=" "$ENV_FILE"
}

# True if the key is in the VISIBLE_KEYS list
is_visible_key() {
    local key="$1" k
    for k in "${VISIBLE_KEYS[@]}"; do
        [[ "$k" == "$key" ]] && return 0
    done
    return 1
}

# Make sure the file ends with a newline before appending
ensure_trailing_newline() {
    if [[ -s "$ENV_FILE" && -n "$(tail -c1 "$ENV_FILE")" ]]; then
        echo >> "$ENV_FILE"
    fi
}

# Prompt for a key's value if it's missing, then append it to .env
prompt_for_key() {
    local key="$1" value=""

    if has_key "$key"; then
        return 0
    fi

    # No one to answer if there's no terminal (CI, cron, piped input)
    if [[ ! -t 0 ]]; then
        echo "Error: $key is missing and no terminal is available to prompt." >&2
        exit 1
    fi

    # Re-prompt until the user enters something
    while [[ -z "$value" ]]; do
        if is_visible_key "$key"; then
            read -r -p "Enter ${key}: " value
        else
            read -r -s -p "Enter ${key}: " value
            echo    # -s suppresses the newline, so print one
        fi
    done

    # Escape backslashes and double quotes so the file stays source-able
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"

    ensure_trailing_newline
    printf '%s="%s"\n' "$key" "$value" >> "$ENV_FILE"
    echo "Saved $key to $ENV_FILE"
}

# --- Main -------------------------------------------------------------------

ensure_env_file

for key in "${REQUIRED_KEYS[@]}"; do
    prompt_for_key "$key"
done

# Optional: load the variables into the current script's environment
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "All required keys are set."

# ...your script continues here...