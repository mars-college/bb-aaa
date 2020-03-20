import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    
    def __init__(self, image_dim=64, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        self.image_dim = image_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(self.image_dim * self.image_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.h_dim, self.z_dim)
        self.fc4 = nn.Linear(self.z_dim, self.h_dim)
        self.fc5 = nn.Linear(self.h_dim, self.image_dim * self.image_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        x = x.view(-1, self.image_dim * self.image_dim)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        x_reconst = x_reconst.view(-1, self.image_dim, self.image_dim)
        return x_reconst, mu, log_var

    



class ConvVAE(nn.Module):
    
    def __init__(self, image_channels=1, z_dim=64):
        super(ConvVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8), 
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16), 
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(128, z_dim)
        self.fc2 = nn.Linear(128, z_dim)
        self.fc3 = nn.Linear(z_dim, 128)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32), 
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16), 
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.view(batch_size, -1))
        z = self.fc3(z)
        x_reconst = self.decoder(z.view(batch_size, 128, 1, 1))
        return x_reconst, mu, logvar
    