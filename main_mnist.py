import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
from syft.workers.websocket_client import WebsocketClientWorker

# setup workers
hook = sy.TorchHook(torch)

w1 = WebsocketClientWorker(host='localhost', hook=hook, id='0', port=8182, log_msgs=True, verbose=True, is_client_worker=False)
w2 = WebsocketClientWorker(host='localhost', hook=hook, id='1', port=8183, log_msgs=True, verbose=True, is_client_worker=False)
w3 = WebsocketClientWorker(host='localhost', hook=hook, id='2', port=8184, log_msgs=True, verbose=True, is_client_worker=False)
w4 = WebsocketClientWorker(host='localhost', hook=hook, id='3', port=8185, log_msgs=True, verbose=True, is_client_worker=False)

hook.local_worker.add_worker(w1)
hook.local_worker.add_worker(w2)
hook.local_worker.add_worker(w3)
hook.local_worker.add_worker(w4)

workers = [w1, w2, w3, w4]



for epoch in range(5):
    for w in workers:
        px, py = w.search('#x')[0], w.search('#y')[0]
        print(w, px, py)
        print('-----')
    time.sleep(2)
