"""
script2.py

Acts as the "doubler" service. It listens for an incoming TCP connection
from script1.py, and every time it receives a number, it doubles it and
sends the result back. Once the doubled value reaches (or exceeds) 100,
it closes the connection.

Run this script FIRST, then run script1.py in another terminal.
"""

import socket

HOST = "127.0.0.1"
PORT = 65432
TARGET = 100


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Allow quick restarts without "Address already in use" errors
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"[script2] Listening on {HOST}:{PORT} ...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"[script2] Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("[script2] Connection closed by peer.")
                    break

                number = int(data.decode())
                doubled = number * 2
                print(f"[script2] Received {number} -> doubling to {doubled}")
                conn.sendall(str(doubled).encode())

                if doubled >= TARGET:
                    print(f"[script2] Reached target ({TARGET}). Stopping.")
                    break


if __name__ == "__main__":
    main()