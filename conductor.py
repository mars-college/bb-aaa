from threading import Thread
import argparse
import os
import json
import time
import asyncio
import websockets
import functools
import queue
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker

class CheckableQueue(queue.Queue): 
    def __contains__(self, item):
        with self.mutex:
            return item in self.queue


class VAE(nn.Module):
    
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    




    
async def handle_worker(websocket, path, hook, workers, update_queue, model, optimizer):
    message = await websocket.recv()
    message = json.loads(message)

    action, name = message['action'], message['name']

    if action == 'register':
        host, syft_port = message['host'], message['syft_port']
        print('register for', name, host, syft_port)
        new_worker = WebsocketClientWorker(host=host, hook=hook, id=name, port=syft_port, log_msgs=True, verbose=True, is_client_worker=False)
        workers[name] = new_worker
        optimizer[workers[name]] = torch.optim.Adam(model.parameters(), lr=1e-3)
    
        message = json.dumps({'success': True})
        await websocket.send(message)

    elif action == 'request_model':
        update_queue.put(name)
        message = json.dumps({'success': True})
        await websocket.send(message)

    elif action == 'check_if_in_queue':
        message = json.dumps({'in_queue': name not in update_queue})
        await websocket.send(message)


async def update_model(loop: asyncio.AbstractEventLoop, workers, model, optimizer, update_queue) -> None:
    while True:       
        print("queue is", update_queue.empty(), update_queue.qsize())
        while not update_queue.empty():
            next_worker_name = update_queue.queue[0]
            worker = workers[next_worker_name]

            print('UPDATE for %s' % next_worker_name, worker)

            model.send(worker)
            x, _ = worker.search('#x')[0], worker.search('#y')[0]
            x_reconst, mu, log_var = model(x)
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # backward
            loss = reconst_loss + kl_div
            optimizer[worker].zero_grad()
            loss.backward()
            optimizer[worker].step()
            model.get()

            print("Reconst Loss: {:.4f}, KL Div: {:.4f}".format(reconst_loss.get(), kl_div.get()))

            update_queue.get()
            
        time.sleep(1)



def start_background_loop(loop: asyncio.AbstractEventLoop, workers, model, optimizer, update_queue) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_model(loop, workers, model, optimizer, update_queue), loop)
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description="Run websocket client.")
    parser.add_argument("--server_port", "-p", type=int, default=8765, help="port number to send messages to conductor, e.g. --msg_port 8765")    
    parser.add_argument("--image_size", "-i", type=int, default=784, help="image size")    
    parser.add_argument("--h_dim", "-d", type=int, default=300, help="neurons in hidden layers")    
    parser.add_argument("--z_dim", "-z", type=int, default=20, help="dimensionality of latent space")    
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="learning rate of optimizer")    
    args = parser.parse_args()
    
    # initialize model and workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(image_size=args.image_size, h_dim=args.h_dim, z_dim=args.z_dim).to(device)
    
    optimizer = {}
    workers, update_queue = {}, CheckableQueue()
    
    # setup update thread
    loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_background_loop, args=(loop, workers, model, optimizer, update_queue), daemon=True)
    update_thread.start()

    # start update thread
    hook = sy.TorchHook(torch)
    start_server = websockets.serve(
        functools.partial(handle_worker, hook=hook, workers=workers, update_queue=update_queue, model=model, optimizer=optimizer),
        '0.0.0.0', args.server_port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__== "__main__":
    main()