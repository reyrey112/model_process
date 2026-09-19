import sqlite3
import time
import threading
import queue

import boto3

import yaml
import redis
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

from util.yaml_check import yaml_key_check

HOSTNAME: str = yaml_key_check(config, "hostname") or "localhost"
REDIS_PORT: int = yaml_key_check(config, "redis_port") or 6379
DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]
STATE_TARGET_STREAM_NAME: str = (
    yaml_key_check(config, "state_target_stream_name") or "state_target"
)
ACTION_STATE_TARGET_STREAM_NAME: str = (
    yaml_key_check(config, "action_state_target_stream_name") or "action_state_target"
)
ACTION_STREAM_NAME: str = yaml_key_check(config, "action_stream_name") or "action"
STATE_COLUMNS: list = config["column_config"]["state_columns"]
ACTION_COLUMN: list = config["column_config"]["action_columns"]
TARGET_COLUMNS: list = config["column_config"]["target_columns"]
ALL_COLUMNS = ACTION_COLUMN + STATE_COLUMNS + TARGET_COLUMNS + ["quality"]
ALL_COLUMNS_SQL = ",\n".join(f'"{col}" REAL' for col in ALL_COLUMNS)


VALUE_COLUMNS_SQL = ",".join(f'"{col}" ' for col in ALL_COLUMNS)
VALUES_SQL = ",".join("?" for col in ALL_COLUMNS)

COLUMN_INDEXES: dict = config["column_config"]["column_indexes"]

r = redis.Redis(
    host=HOSTNAME, port=REDIS_PORT, decode_responses=True, password="reyden"
)

# internal thread safe queue
data_queue = queue.Queue()

from app.backend import api_client as api


def init_local_db():
    conn = sqlite3.connect("process.db", check_same_thread=False)
    cur = conn.cursor()

    # optimizations
    cur.execute("PRAGMA journal_model=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # create table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS process_data (
            Timestamp REAL PRIMARY KEY,
            {ALL_COLUMNS_SQL}
            )
    """)
    conn.commit()
    return conn, cur


def log_data_local(conn, cur, action_state_target_dict: dict):
    ts = time.time()
    all_columns = ["Timestamp"] + ALL_COLUMNS

    values_list = [action_state_target_dict[x] for x in all_columns]

    # use deque instead if large?
    # move timestam from backj to front of list
    values_list.insert(0, values_list.pop())

    value_names_sql = ",".join(f'"{col}" ' for col in all_columns)
    value_holders_sql = ",".join("?" for col in all_columns)

    statement = f"""INSERT OR REPLACE INTO process_data ({value_names_sql}) VALUES ({value_holders_sql})"""

    values_sql = tuple(values_list)

    cur.execute(
        statement,
        values_sql,
    )

    conn.commit()

    # pass to background clud syncer

    data_dict = {all_columns[x]: values_list[x] for x in range(len(all_columns))}

    data_queue.put(data_dict)


def cloud_sync_worker():
    """Background thread that makes micro-batches for cloud upload"""

    # initilize AWS Timestream Clint
    # ts_client = boto3.client("timestream-write", region_name="us-east-1")

    # ensuring correct schema is used
    all_columns = ["Timestamp"] + ALL_COLUMNS
    value_names_sql = ",".join(f'"{col}" ' for col in all_columns)
    value_holders_sql = ",".join("?" for col in all_columns)


    batch = []

    # pull items out of queue, block if empty
    item = data_queue.get()
    batch.append(item)

    while len(batch) < 1000:
        try:
            batch.append(data_queue.get_nowait())
        except queue.Empty:
            break

    # convert batch to list of tuples for executemany

    data_tuples = []

    for i in batch:
        data = tuple(i[x] for x in all_columns)
        data_tuples.append(data)

    success = False
    sec = 2
    print(f"inputting {len(data_tuples)} into db")

    while not success:
        try:
            print(f"inputting {len(data_tuples)} into db")
            response = api.db_write(data_tuples=data_tuples, all_columns=all_columns)
            if response["status"] == "success":
                print(f"Successfully inserted {response["inserted_records"]} into db")

                for _ in range(len(batch)):
                    data_queue.task_done()

                success = True

        except Exception as e:
            print(f"Network error: {e}. Retrying in {sec} seconds")
            time.sleep(sec)
            sec *= 2


# --- Initialize and Run ---
conn, cur = init_local_db()

# Start background cloud syncing
threading.Thread(target=cloud_sync_worker, daemon=True).start()
count = 0
try:
    print("Starting data ingestion loop at 200 Hz...")
    while True:
        # runs whenevr new redis data is recieveed
        action_state_target = r.xread(
            block=10000, streams={ACTION_STATE_TARGET_STREAM_NAME: "$"}
        )
        action_state_target_dict = action_state_target[0][1][0][1]
        log_data_local(conn, cur, action_state_target_dict)
        print(f"logged in DB and queue: {count}")
        count += 1
except KeyboardInterrupt:
    print("Shutting down cleanly.")
finally:
    conn.close()
