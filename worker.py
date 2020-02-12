from threading import Thread
from socket import gethostbyname_ex, gethostname
from enum import Enum
from sys import platform
import argparse
import time
import asyncio
import websockets
import os
import json
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

if platform == "linux" or platform == "linux2":
    from pyroute2 import IPRoute
    ip = IPRoute()
    local_ip_address = dict(ip.get_addr(label='eth0')[0]['attrs'])['IFA_LOCAL']
elif platform == "darwin":
    from socket import gethostbyname_ex, gethostname
    local_ip_address = gethostbyname_ex(gethostname())[-1][-1]


class Worker:

    class Mode(Enum):
        CAPTURING = 1
        READY_TO_UPDATE = 2
        WAITING = 3

    def __init__(self, name, conductor_ip, conductor_port, syft_port, batch_size, hook):
        self.name = name
        self.conductor_ip = conductor_ip
        self.conductor_port = conductor_port
        self.syft_port = syft_port
        self.batch_size = batch_size
        self.hook = hook

        self.registered = False
        self.active = False

        self.captures = []
        self.n_updates = 0
        self.socket_uri = 'ws://%s:%d' % (conductor_ip, conductor_port)

    def set_mode(self, mode):
        self.mode = mode
            
    def setup_data(self, image_size):
        self.image_size = image_size
        trans = transforms.ToTensor() #transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        mnist_loader = torch.utils.data.DataLoader(
            dataset=datasets.MNIST(root='./data', train=True, transform=trans, download=True), 
            batch_size=self.batch_size, shuffle=True)
        self.mnist_iterator = iter(mnist_loader)

    def ready_to_update(self):
        ready = (self.num_captures() == self.batch_size)
        return ready

    def get_next_batch(self):
        print('Get next batch of mnist')
        x, y = next(self.mnist_iterator)
        x = x.view(-1, self.image_size)
        self.x_ptr = x.tag('#x').send(self.local_worker)
        self.y_ptr = y.tag('#y').send(self.local_worker)
        
    def activate(self):
        kwargs = { "hook": self.hook,
            "id": self.name,
            "host": "0.0.0.0",
            "port": self.syft_port }
        self.active = True
        self.local_worker = WebsocketServerWorker(**kwargs)
        self.local_worker.start()

    async def capture(self):
        print('run the camera')
        image = None
        self.captures.append(image)

    def num_captures(self):
        return len(self.captures)

    async def try_register(self):
        print('Register with conductor')
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'register', 'name': self.name, 'syft_port': self.syft_port, 'host': local_ip_address})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            if result['success']:
                print('Registration successful.')
                self.registered = True

    async def request_update(self):
        print('Request to make a model update')
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'request_model', 'name': self.name})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            if result['success']:
                print('ready to make an update...')
    
    async def check_if_in_queue(self):
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'check_if_in_queue', 'name': self.name})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            return result['in_queue']



async def update_worker(loop: asyncio.AbstractEventLoop, worker) -> None:

    while not worker.registered:

        await worker.try_register()
        time.sleep(1)

    while worker.active:

        if worker.mode == Worker.Mode.CAPTURING:
            await worker.capture()
            if worker.ready_to_update():
                worker.set_mode(Worker.Mode.READY_TO_UPDATE)

        elif worker.mode == Worker.Mode.READY_TO_UPDATE:
            worker.get_next_batch()
            await worker.request_update()
            worker.set_mode(Worker.Mode.WAITING)

        elif worker.mode == Worker.Mode.WAITING:
            done = await worker.check_if_in_queue()
            if done:
                worker.n_updates += 1
                worker.captures = []
                worker.set_mode(Worker.Mode.CAPTURING)

        time.sleep(1)
    

def start_background_loop(loop: asyncio.AbstractEventLoop, worker) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_worker(loop, worker), loop)
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument("--name", type=str, help="name of the worker, e.g. worker001")
    parser.add_argument("--conductor_ip", type=str, default="0.0.0.0", help="host for the socket connection")
    parser.add_argument("--conductor_port", type=int, default=8765, help="port number of the websocket server worker, e.g. --conductor_port 8765")
    parser.add_argument("--syft_port", type=int, help="port for syft worker")
    parser.add_argument("--batch_size", type=int, default=8, help="number of batches")
    parser.add_argument("--verbose", "-v", action="store_true", help="if set, websocket server worker will be started in verbose mode")
    
    args = parser.parse_args()
    
    # setup worker
    hook = sy.TorchHook(torch)
    worker = Worker(args.name, args.conductor_ip, args.conductor_port, args.syft_port, args.batch_size, hook)    
    worker.setup_data(784)
    worker.set_mode(Worker.Mode.CAPTURING)

    # setup update thread
    loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_background_loop, args=(loop, worker,), daemon=True)
    update_thread.start()

    # launch syft worker
    worker.activate()

    
if __name__== "__main__":
    main()