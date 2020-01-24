import torch
import syft
from syft.workers.websocket_server import WebsocketServerWorker
from syft.workers.websocket_client import WebsocketClientWorker

hook = syft.TorchHook(torch)

#local_worker = WebsocketServerWorker(host='localhost', hook=hook, id=0, port=8182, log_msgs=True, verbose=True)

#hook = syft.TorchHook(torch, local_worker=local_worker)


remote_client = WebsocketClientWorker(host='localhost', hook=hook, id=2, port=8182, log_msgs=True, verbose=True)

hook.local_worker.add_worker(remote_client)


y = torch.tensor([1,2,3,2,4])
y.send(remote_client)


x = torch.tensor([3,1,5,-1,0,44])
x.send(remote_client)
