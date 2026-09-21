# upload_env_to_ssm.py
import boto3
import os
import sys

# --- Config ---
ENV_FILE = ".env"
SSM_PREFIX = "/model_process/prod"  
SENSITIVE_KEYS = {
    "SECRET_KEY",
    "AWS_STORAGE_BUCKET_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "DJANGO_POSTGRES_USER",
    "DJANGO_POSTGRES_PASSWORD",
    "DJANGO_POSTGRES_DB",
    "MLFLOW_POSTGRES_USER",
    "MLFLOW_POSTGRES_PASSWORD",
    "MLFLOW_POSTGRES_DB",
    "MYSQL_ROOT_PASSWORD",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
    "ZENML_STORE_USERNAME",
    "ZENML_STORE_PASSWORD",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MEDIA_URL",
    "GITHUB_PAT"
}
# --------------


def parse_env_file(filepath):
    variables = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                variables[key.strip()] = value.strip().strip('"').strip("'")
    return variables


def upload_to_ssm(variables, prefix, sensitive_keys):
    if os.getenv('GITHUB_ACTIONS'):
        session = boto3.Session()
    else:
        session = boto3.Session(profile_name="admin")

    client = session.client(
        "ssm",
    )
    success, failed = [], []

    for key, value in variables.items():
        param_type = "SecureString" if key in sensitive_keys else "String"
        param_name = f"{prefix}/{key}"

        try:
            client.put_parameter(
                Name=param_name,
                Value=value,
                Type=param_type,
                Overwrite=True,
            )
            print(f"  ✓ {param_name} ({param_type})")
            success.append(key)
        except Exception as e:
            print(f"  ✗ {param_name} — Error: {e}")
            failed.append(key)

    return success, failed


def main():
    if not os.path.exists(ENV_FILE):
        print(f"Error: {ENV_FILE} not found")
        sys.exit(1)

    print(f"Reading {ENV_FILE}...")
    variables = parse_env_file(ENV_FILE)
    print(f"Found {len(variables)} variables\n")

    print(f"Uploading to SSM under prefix '{SSM_PREFIX}'...")
    success, failed = upload_to_ssm(variables, SSM_PREFIX, SENSITIVE_KEYS)

    print(f"\nDone: {len(success)} uploaded, {len(failed)} failed")
    if failed:
        print(f"Failed keys: {', '.join(failed)}")


if __name__ == "__main__":
    main()