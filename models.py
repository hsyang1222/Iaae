import torch
import numpy as np
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(64, latent_dim)
        self.sigma = nn.Linear(64, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        mu, sigma = self.encode(x_flat)
        z_posterior = self.reparameterize(mu, sigma)
        return z_posterior

    def encode(self, x):
        x = self.model(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        from torch.autograd import Variable
        batch_size = mu.size(0)
        eps = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))).cuda()
        return eps * sigma + mu


class Decoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, int(np.prod(image_shape))),
            nn.Tanh(),
        )
        self.image_shape = image_shape

    def forward(self, z_posterior):
        decoded_flat = self.model(z_posterior)
        decoded = decoded_flat.view(decoded_flat.shape[0], *self.image_shape)
        return decoded


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

#################################################################################################
# 양현식


# 김상엽


# 한누리

if __name__ == '__main__':
    img_size = 32
    latent_dim = 10
    image_shape = [3, img_size, img_size]

    encoder = Encoder(latent_dim, image_shape)
    decoder = Decoder(latent_dim, image_shape)
    discriminator = Discriminator(latent_dim)