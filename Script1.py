"""
script1.py

Acts as the "initiator". It connects to script2.py, sends a starting
number, receives the doubled result, and sends that result right back
so it can be doubled again. This continues until the value reaches
(or exceeds) 100.

Run script2.py FIRST (so it's listening), then run this script.
"""

import socket

HOST = "127.0.0.1"
PORT = 65432
TARGET = 100
START_NUMBER = 1  # change this to start from a different number


def main():
    number = START_NUMBER
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        print(f"[script1] Connected to script2 at {HOST}:{PORT}")

        while True:
            print(f"[script1] Sending {number}")
            client_socket.sendall(str(number).encode())

            data = client_socket.recv(1024)
            if not data:
                print("[script1] Connection closed by peer.")
                break

            number = int(data.decode())
            print(f"[script1] Received {number} back from script2")

            if number >= TARGET:
                print(f"[script1] Reached target ({TARGET}). Stopping.")
                break


if __name__ == "__main__":
    main()