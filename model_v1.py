import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy



class VAE(nn.Module):

    def __init__(self,):

       super(VAE, self).__init__()
       self.fc1 = nn.Linear(60,256)
       self.fc21 = nn.Linear(256,10)
       self.fc22 = nn.Linear(256,10)
       self.fc3 = nn.Linear(10,256)
       self.fc4 = nn.Linear(256,60)

    def encode(self, x):
#        print("shape of fc1 VAE is", numpy.shape(self.fc1(x)),"on", numpy.shape(x))
        h1 = F.relu(self.fc1(x))
#        print("h1 shape is", numpy.shape(h1), "fc2 makes it", numpy.shape(self.fc21(h1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
