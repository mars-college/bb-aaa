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

from PIL import Image
import numpy as np


# setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# model & training parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 50
batch_size = 1
learning_rate = 1e-3


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

    



# setup workers
hook = sy.TorchHook(torch)

w1 = WebsocketClientWorker(host='10.79.10.219', hook=hook, id='0', port=8182, log_msgs=True, verbose=True, is_client_worker=False)
w2 = WebsocketClientWorker(host='localhost', hook=hook, id='1', port=8183, log_msgs=True, verbose=True, is_client_worker=False)
w3 = WebsocketClientWorker(host='localhost', hook=hook, id='2', port=8184, log_msgs=True, verbose=True, is_client_worker=False)
w4 = WebsocketClientWorker(host='localhost', hook=hook, id='3', port=8185, log_msgs=True, verbose=True, is_client_worker=False)

hook.local_worker.add_worker(w1)
hook.local_worker.add_worker(w2)
hook.local_worker.add_worker(w3)
hook.local_worker.add_worker(w4)

workers = [w1, w2, w3, w4]

model = VAE().to(device)



optimizer = {}

for w in workers:
    optimizer[w] = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for w, worker in enumerate(workers):
        
        print('epoch', epoch, 'worker', worker.id)
        
        model.send(worker)
        x, y = worker.search('#x')[0], worker.search('#y')[0]

        #x, y = x.to(device).view(-1, image_size), y.to(device)

        #model.send(px.location) 
        print('-----')

        x_reconst, mu, log_var = model(x)

        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # backward
        loss = reconst_loss + kl_div
        optimizer[worker].zero_grad()
        loss.backward()
        optimizer[worker].step()

        model.get()
        
        z = x_reconst.get()
        Image.fromarray(np.clip(255*z.reshape([28,28]).detach().numpy(), 0, 255).astype(np.uint8)).convert('RGB').save('myimage%02d_%02d.png'%(epoch, w))

    
    print ("Epoch[{}/{}], Worker[{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
       .format(epoch+1, num_epochs, w+1, len(workers), reconst_loss.get(), kl_div.get()))

    








# for epoch in range(num_epochs):
#     for w, worker in enumerate(workers):
#         print('epoch', epoch, 'worker', worker.id)
#         model.send(worker)
#         x, y = worker.search('#x')[0], worker.search('#y')[0]
#         x, y = x.to(device).view(-1, image_size), y.to(device)
#         x_reconst, mu, log_var = model(x)
#         reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
#         kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         # backward
#         loss = reconst_loss + kl_div
#         optimizer[worker].zero_grad()
#         loss.backward()
#         optimizer[worker].step()
#         model.get()

    