from threading import Thread
import argparse
import json
from json import JSONEncoder
import time
import asyncio
import websockets
import functools
import queue
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from models import VAE, ConvVAE


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class CheckableQueue(queue.Queue): 
    def __contains__(self, item):
        with self.mutex:
            return item in self.queue


async def handle_worker(websocket, path, hook, workers, update_queue, results_box, model, optimizer, learning_rate, verbose=False):
    message = await websocket.recv()
    message = json.loads(message)

    action, name = message['action'], message['name']

    if action == 'register':
        host, syft_port = message['host'], message['syft_port']
        new_worker = WebsocketClientWorker(host=host, hook=hook, id=name, port=syft_port, log_msgs=True, verbose=verbose, is_client_worker=False)
        workers[name] = new_worker
        optimizer[workers[name]] = torch.optim.Adam(model.parameters(), lr=learning_rate)
        message = json.dumps({'success': True})
        print('Worker %s (%s) registered, port %d' % (name, host, syft_port))
        await websocket.send(message)

    elif action == 'request_model':
        update_queue.put(name)
        message = json.dumps({'success': True})
        await websocket.send(message)

    elif action == 'ping_conductor':
        result = {'in_queue': name not in update_queue}
        if not result['in_queue'] and name in results_box:
            result['user'] = results_box[name]
        message = json.dumps(result, cls=NumpyArrayEncoder)
        await websocket.send(message)


async def update_model(loop: asyncio.AbstractEventLoop, workers, model, optimizer, image_dim, update_queue, results_box) -> None:
    batch = 1
    model.train()

    while True:
        
        while not update_queue.empty():

            worker_name = update_queue.queue[0]
            print('Update requested from %s' % worker_name)

            # send model to the worker
            worker = workers[worker_name]
            model.send(worker)
            
            # forward pass with pointer to batch from worker
            x = worker.search('#x')[0]
            x_reconst, mu, log_var = model(x)
            
            # get loss
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # backward pass
            loss = reconst_loss + kl_div
            optimizer[worker].zero_grad()
            loss.backward()
            optimizer[worker].step()
            reconst_loss_f, kl_div_f = reconst_loss.get().detach(), kl_div.get().detach()
            print("Reconstruction Loss: {:.4f}, KL Divergence: {:.4f}".format(reconst_loss_f, kl_div_f))

            # draw first reconstructed sample 
            #  note: should be done by client
            reconstructed_samples = x_reconst.get()
            batch_size = reconstructed_samples.shape[0]
            reconstructed_samples = reconstructed_samples.reshape([batch_size, image_dim, image_dim])
            reconstructed_samples = (255 * reconstructed_samples.detach().numpy()).astype(np.uint8)
            Image.fromarray(reconstructed_samples[0]).save('reconstructed_%03d.png' % batch)
            results_box[worker_name] = {'success': True, 'reconstruction_loss': reconst_loss_f.item(), 'kl_loss': kl_div_f.item(), 'reconstructed_image': reconstructed_samples[0], 'random_image': reconstructed_samples[0]}

            # get back the model, update queue
            model.get()
            update_queue.get()

            batch += 1

        time.sleep(1)


def start_background_loop(loop: asyncio.AbstractEventLoop, workers, model, optimizer, image_dim, update_queue, results_box) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_model(loop, workers, model, optimizer, image_dim, update_queue, results_box), loop)
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description="Run conductor application")
    parser.add_argument("--server_ip", type=str, default='0.0.0.0', help="IP address of conductor (default 0.0.0.0)")
    parser.add_argument("--server_port", type=int, default=8765, help="port number to send messages to conductor, e.g. --msg_port 8765")    
    #parser.add_argument("--image_nc", type=int, default=1, help="number of channels")    
    parser.add_argument("--plain_vae", action="store_true", help="if set, uses simple fully-connected VAE, otherwise uses ConvVAE")
    parser.add_argument("--image_dim", type=int, default=64, help="image size")    
    parser.add_argument("--h_dim", type=int, default=300, help="neurons in hidden layers")    
    parser.add_argument("--z_dim", type=int, default=20, help="dimensionality of latent space")    
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate of optimizer")    
    parser.add_argument("--verbose", action="store_true", help="if set, websocket server worker will be started in verbose mode")
    args = parser.parse_args()
    
    # initialize model and workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
    if args.plain_vae:
        print('Making a plain fully-connected VAE with h-dimension=%d, z-dimension=%d' % (args.h_dim, args.z_dim))
        model = VAE(image_dim=args.image_dim, h_dim=args.h_dim, z_dim=args.z_dim).to(device)
    else:
        print('Making a convolutional VAE with z-dimension=%d' % args.z_dim)
        model = ConvVAE(z_dim=args.z_dim).to(device)
    
    optimizer = {}
    workers, results_box, update_queue = {}, {}, CheckableQueue()
    
    # setup update thread
    loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_background_loop, args=(loop, workers, model, optimizer, args.image_dim, update_queue, results_box), daemon=True)
    update_thread.start()

    # start update thread
    hook = sy.TorchHook(torch)
    start_server = websockets.serve(
        functools.partial(handle_worker, hook=hook, workers=workers, update_queue=update_queue, results_box=results_box, model=model, optimizer=optimizer, learning_rate=args.learning_rate, verbose=args.verbose),
        args.server_ip, args.server_port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
    

if __name__== "__main__":
    main()