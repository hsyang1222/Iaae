import torch
import numpy as np
import torch.nn as nn
import math
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, nz, image_size):
        super(Encoder, self).__init__()
        nc = 3
        nf=32
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=4*4*nf*4, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mu = nn.Linear(in_features=1024, out_features=nz)
        #self.var = nn.Linear(in_features=1024, out_features=nz)

    def forward(self, input):
        y =  self.net(input)
        return self.mu(y)#, self.var(y)

class Mapping(nn.Module) : 
    def __init__(self, nz, linear_num) : 
        super(Mapping, self).__init__()
        
        linear = nn.ModuleList()
        for i in range(linear_num) : 
            linear.append( nn.Linear(in_features=nz, out_features=nz) )
            linear.append( nn.ELU() )
        self.linear = linear
        
    def forward(self, input):
        for layer in self.linear : 
            input = layer(input)
        return input
        
        

class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(len(input), self.shape[0], self.shape[1], self.shape[2])
    
class Decoder(nn.Module):
    def __init__(self, nz, image_size, linear_add=0):
        nc = 3
        nf= 32
        super(Decoder, self).__init__()
        
        linear = nn.ModuleList()
        for i in range(linear_add) : 
            linear.append( nn.Linear(in_features=nz, out_features=nz) )
            linear.append( nn.ReLU() )
        self.linear = linear
        
        self.net = nn.Sequential(
            nn.Linear(in_features=nz, out_features=4*4*nf*4),
            Unflatten((128, 4, 4)),
            
            # add output_padding=1 to ConvTranspose2d to reconstruct original size
            nn.ConvTranspose2d(nf*4, nf*2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf*2, nf, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf, int(nf/2), 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(int(nf/2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int(nf/2), nc, 5, 1, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        for layer in self.linear : 
            input = layer(input)
        
        return self.net(input)


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
class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            #out = fused_leaky_relu(out, self.bias * self.lr_mul)
            out = F.threshold(input, self.threshold, self.value, self.inplace)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

class WeakNorm(nn.Module):
    def __init__(self) :
        super(WeakNorm, self).__init__()
        # sigmoid(v)=1 then x' = x / Var(x)
        # sigmoid(v)=0 then x' = x
        #self.v = torch.tensor([-3.], requires_grad=True)
        self.v = torch.nn.parameter.Parameter(torch.tensor([-3.], requires_grad=True))
    
    def forward(self, x) : 
        sig_v = torch.sigmoid(self.v)
        var_x = torch.var(x)
        x = x / (1-sig_v) * var_x + sig_v * 1
        return x

class ModEncoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(ModEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            WeakNorm(),
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
        eps = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))).to(mu.device)
        return eps * sigma + mu


class ModDecoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(ModDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            #nn.BatchNorm1d(128),
            WeakNorm(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, int(np.prod(image_shape))),
            nn.Tanh(),
        )
        self.image_shape = image_shape

    def forward(self, z_posterior):
        decoded_flat = self.model(z_posterior)
        decoded = decoded_flat.view(decoded_flat.shape[0], *self.image_shape)
        return decoded

class Mod2Encoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Mod2Encoder, self).__init__()
        self.model = nn.Sequential(
            EqualLinear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(128, 64),
            #nn.BatchNorm1d(64),
            #WeakNorm(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = EqualLinear(64, latent_dim)
        self.sigma = EqualLinear(64, latent_dim)
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
        eps = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))).to(mu.device)
        return eps * sigma + mu


class Mod2Decoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Mod2Decoder, self).__init__()

        self.model = nn.Sequential(
            EqualLinear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(64, 128),
            #nn.BatchNorm1d(128),
            #WeakNorm(),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(128, int(np.prod(image_shape))),
            nn.Tanh(),
        )
        self.image_shape = image_shape

    def forward(self, z_posterior):
        decoded_flat = self.model(z_posterior)
        decoded = decoded_flat.view(decoded_flat.shape[0], *self.image_shape)
        return decoded


class StackDecoder(nn.Module):
    def __init__(self, nz, image_size):
        nc = 3
        nf= 32
        latent_dim = nz
        super(StackDecoder, self).__init__()
        self.net = nn.Sequential(
            #mapping 
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            ###
            
            nn.Linear(in_features=nz, out_features=4*4*nf*4),
            nn.BatchNorm1d(num_features=4*4*nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            Unflatten((128, 4, 4)),
            
            # add output_padding=1 to ConvTranspose2d to reconstruct original size
            nn.ConvTranspose2d(nf*4, nf*2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf*2, nf, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nf, int(nf/2), 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(int(nf/2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int(nf/2), nc, 5, 1, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.net(input)    
    
    
    
    
    
    
    
    
class GME_Encoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(GME_Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(64, latent_dim)
        #self.sigma = nn.Linear(64, latent_dim)
        self.cov = nn.Linear(64, latent_dim*latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        mu, cov = self.encode(x_flat)
        z_posterior = self.reparameterize(mu, cov)
        return z_posterior

    def encode(self, x):
        x = self.model(x)
        mu = self.mu(x)
        cov_flatten = self.cov(x)
        return mu, cov_flatten.view(-1,self.latent_dim,self.latent_dim)

    def reparameterize(self, mu, cov):
        device = mu.device
        Sigma_k = torch.matmul(cov, cov.permute([0,2,1]))
        Sigma_k.add_(torch.eye(self.latent_dim).to(device))
        self.MGM = torch.distributions.multivariate_normal.MultivariateNormal(mu, Sigma_k)
        random_value = self.MGM.sample()
        
        #eps = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))).to(device)
        return random_value


class GME_Decoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(GME_Decoder, self).__init__()

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


class GME_Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(GME_Discriminator, self).__init__()
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z.view(z.size(0),-1))
        return validity    
    
class DirectEncoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(DirectEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        x = self.model(x_flat)
        return x    
    
    
class ImageDiscriminator(nn.Module):
    def __init__(self, image_shape):
        super(ImageDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 64),
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

    def forward(self, x):
        x_flatten = x.view(x.shape[0], -1)
        validity = self.model(x_flatten)
        return validity
# 김상엽


# 한누리

if __name__ == '__main__':
    img_size = 32
    latent_dim = 10
    image_shape = [3, img_size, img_size]

    encoder = Encoder(latent_dim, image_shape)
    decoder = Decoder(latent_dim, image_shape)
    discriminator = Discriminator(latent_dim)