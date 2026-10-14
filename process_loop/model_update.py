from dotenv import load_dotenv,find_dotenv
import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

load_dotenv(find_dotenv())
HOSTNAME = os.environ.get("HOSTNAME", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")


def main ():
    r = redis.Redis(
        host=HOSTNAME, port=int(REDIS_PORT), decode_responses=True, password=REDIS_PASSWORD
    )

