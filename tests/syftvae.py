import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import syft as sy

# bob & alice
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# model & training parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 10
batch_size = 64
learning_rate = 1e-3

# data loader
federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
    torchvision.datasets.MNIST(root='../../data', 
                   train=True, download=True, transform=transforms.ToTensor())
    .federate((bob, alice)), 
    batch_size=batch_size, shuffle=True, **kwargs)


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

    
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(federated_train_loader): 
        model.send(data.location) 
        
        data, target = data.to(device).view(-1, image_size), target.to(device)
        x_reconst, mu, log_var = model(data)

        reconst_loss = F.binary_cross_entropy(x_reconst, data, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # backward
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.get()

        if (batch_idx+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, batch_idx+1, len(federated_train_loader), reconst_loss.get(), kl_div.get()))
    
