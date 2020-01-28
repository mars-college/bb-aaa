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

#bob = sy.VirtualWorker(hook, id="bob")
#alice = sy.VirtualWorker(hook, id="alice")

rc1 = WebsocketClientWorker(host='localhost', hook=hook, id='0', port=8182, log_msgs=True, verbose=True, is_client_worker=False)
rc2 = WebsocketClientWorker(host='localhost', hook=hook, id='1', port=8183, log_msgs=True, verbose=True, is_client_worker=False)
rc3 = WebsocketClientWorker(host='localhost', hook=hook, id='2', port=8184, log_msgs=True, verbose=True, is_client_worker=False)
rc4 = WebsocketClientWorker(host='localhost', hook=hook, id='3', port=8185, log_msgs=True, verbose=True, is_client_worker=False)

# rc1 = sy.VirtualWorker(hook, id="0")
# rc2 = sy.VirtualWorker(hook, id="1")
# rc3 = sy.VirtualWorker(hook, id="2")
# rc4 = sy.VirtualWorker(hook, id="3")

hook.local_worker.add_worker(rc1)
hook.local_worker.add_worker(rc2)
hook.local_worker.add_worker(rc3)
hook.local_worker.add_worker(rc4)



rc1px, rc1py = rc1.search('#mydata')[0], rc1.search('#mydata')[0]
rc1d = sy.BaseDataset(rc1px, rc1py)

rc2px, rc2py = rc2.search('#mydata')[0], rc2.search('#mydata')[0]
rc2d = sy.BaseDataset(rc2px, rc2py)

rc3px, rc3py = rc3.search('#mydata')[0], rc3.search('#mydata')[0]
rc3d = sy.BaseDataset(rc3px, rc3py)

rc4px, rc4py = rc4.search('#mydata')[0], rc4.search('#mydata')[0]
rc4d = sy.BaseDataset(rc4px, rc4py)


federated_dataset = sy.FederatedDataset([rc1d, rc2d, rc3d, rc4d])
for data, target in sy.FederatedDataLoader(federated_dataset, batch_size=8):
    print(data)