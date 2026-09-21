import boto3
import os

SSM_PREFIX = "/model_process/prod"
OUTPUT_FILE = ".env"

def download_from_ssm(prefix, output_file):
    client = boto3.client("ssm")
    paginator = client.get_paginator("get_parameters_by_path")

    variables = {}
    for page in paginator.paginate(Path=prefix, WithDecryption=True):
        for param in page["Parameters"]:
            key = param["Name"].replace(f"{prefix}/", "")
            variables[key] = param["Value"]

    with open(output_file, "w") as f:
        for key, value in variables.items():
            f.write(f'{key}="{value}"\n')
    os.chmod(output_file, 0o600)  # restrict file permissions
    print(f"Written {len(variables)} variables to {output_file}")

if __name__ == "__main__":
    download_from_ssm(SSM_PREFIX, OUTPUT_FILE)