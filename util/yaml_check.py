import yaml


def yaml_add_key(yaml_file_path, key: str, value):
    # 1. Read the existing data
    with open(yaml_file_path, "r") as file:
        current_data = yaml.safe_load(file)

    # 2. Modify the data (append to the inner list)
    current_data[key] = value

    # 3. Write the entire updated data back to the file
    with open(yaml_file_path, "w") as file:
        yaml.safe_dump(current_data, file, default_flow_style=False, sort_keys=False)


def yaml_update_key(yaml_file_path, key: str, value):
    # 1. Read the existing data
    with open(yaml_file_path, "r") as file:
        current_data = yaml.safe_load(file)

    # 2. Update the specific keys
    current_data[key] = value

    # 3. Write the updated data back
    with open(yaml_file_path, "w") as file:
        yaml.safe_dump(current_data, file, default_flow_style=False, sort_keys=False)


def yaml_add_or_update(key: str, value, yaml_file_path="config.yaml"):
    with open(yaml_file_path, "r") as file:
        current_data = yaml.safe_load(file) or {}
    if key in current_data:
        yaml_update_key(yaml_file_path, key, value)
    elif not current_data:
        with open("config.yaml", "w") as file:
            pass
        yaml_add_key(yaml_file_path, key, value)
    else:
        yaml_add_key(yaml_file_path, key, value)
