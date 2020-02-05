import argparse
import time
from threading import Thread
import asyncio
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

# setup mnist
use_cuda = torch.cuda.is_available()
trans = transforms.ToTensor() #transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
mnist = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
mnist_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=1, shuffle=True)
mnist_iterator = iter(mnist_loader)


# setup syft
hook = sy.TorchHook(torch)

local_worker = None
x_ptr, y_ptr = None, None

image_size = 784
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_next(hook, local_worker):
    global x_ptr, y_ptr
    x, y = next(mnist_iterator)
    x = x.view(-1, image_size)
    x_ptr = x.tag('#x').send(local_worker)
    y_ptr = y.tag('#y').send(local_worker)


async def update_worker(loop: asyncio.AbstractEventLoop) -> None:
    while True:
        get_next(hook, local_worker)
        #print(local_worker._objects)
        time.sleep(5)


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

    kwargs = {
        "id": str(args.id),
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    } 

    global local_worker
    local_worker = WebsocketServerWorker(**kwargs)
    loop = asyncio.new_event_loop()

    update_thread = Thread(target=start_background_loop, args=(loop,), daemon=True)
    update_thread.start()

    local_worker.start()      


if __name__== "__main__":
    main()