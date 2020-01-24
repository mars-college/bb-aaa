import argparse
import time
from threading import Thread
import asyncio
import torch
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker


async def update_worker(loop: asyncio.AbstractEventLoop) -> None:
    while True:
        # do stuff here
        time.sleep(1)


def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_worker(loop), loop)
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument("--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777")
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument("--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice")
    parser.add_argument("--verbose", "-v", action="store_true", help="if set, websocket server worker will be started in verbose mode")
    args = parser.parse_args()

    hook = sy.TorchHook(torch)

    kwargs = {
        "id": str(args.id),
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    } 

    global local_worker
    local_worker = WebsocketServerWorker(**kwargs)
    local_worker.start()  
    
    loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_background_loop, args=(loop,), daemon=True)
    update_thread.start()


if __name__== "__main__":
    main()