import torch
import numpy as np
import torch.nn as nn
import math
from torch.nn import functional as F

class MimicSingle(nn.Module) :
    def __init__(self, feature) : 
        super(Mimic, self).__init__()

        self.net = torch.nn.Sequential(
                            nn.Linear(feature, 200),
                            nn.CELU(),
                            nn.Linear(200, 300),
                            nn.CELU(),
                            nn.Linear(300, 400),
                            nn.CELU(),
                            nn.Linear(400, feature)
                            )

    def forward(self, x) : 
        predict = self.net(x)
        return predict
    
class MimicStack(nn.Module) :
    def __init__(self, in_feature, out_feature, hidden_feature, hidden_layer) : 
        super(MimicStack, self).__init__()
        
        linear_list = []
        
        linear_list.append(nn.Linear(in_feature, hidden_feature))
        linear_list.append(nn.CELU())
        
        
        
        for i in range(hidden_layer) : 
            linear_list.append(nn.Linear(hidden_feature, hidden_feature))
            linear_list.append(nn.CELU())
            
        linear_list.append(nn.Linear(hidden_feature, out_feature))
        
        self.net = torch.nn.ModuleList(linear_list)
        
    def forward(self, x) : 
        for layer in self.net :
            x = layer(x)

        return x

class Mimic(nn.Module) : 
    def __init__(self, in_feature, out_feature, hidden_feature, hidden_layer) :
        super(Mimic, self).__init__()
        
        assert in_feature == out_feature
        
        def single_line(hidden_feature, hidden_layer) : 
            linear_list = []

            linear_list.append(nn.Linear(1, hidden_feature))
            linear_list.append(nn.CELU())
        
            for i in range(hidden_layer) : 
                linear_list.append(nn.Linear(hidden_feature, hidden_feature))
                linear_list.append(nn.CELU(True))

            linear_list.append(nn.Linear(hidden_feature, 1))
        
            return torch.nn.ModuleList(linear_list)
        
        net_list = [single_line(hidden_feature, hidden_layer) for i in range(in_feature)] 
        self.net_list = torch.nn.ModuleList(net_list)
        
    def forward(self, x) :
        assert x.size(1) == len(self.net_list)
        
        predict_list = []
        
        for each_dim in range(x.size(1)) :
            x_each_dim = x[:,each_dim].view(-1,1)
            net_each_dim = self.net_list[each_dim]
            
            predict_each_dim = x_each_dim
            for each_layer in net_each_dim : 
                predict_each_dim = each_layer(predict_each_dim)
            
            predict_list.append(predict_each_dim)
            
        return torch.cat(predict_list, dim=1)
        
    
    
    
class Encoder(nn.Module):
    
    def __init__(self, nz=32, img_size=32, ngpu=1, ndf=64, ngf=64, nc=3):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        
        kernel = 2*(img_size//32)
        # img_size 32 --> 2
        # img_size 64 --> 4
        # img_size 128 --> 8
        # img_size 256 --> 16
        
        #print(kernel)
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, nz, kernel, 1, 0, bias=False),
            # state size. 1x1x1
        )
        
        self.nz = nz

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, self.nz)

    

class Mapping(nn.Module) : 
    def __init__(self, in_out_nz, nz, linear_num) : 
        super(Mapping, self).__init__()
        
        linear = nn.ModuleList()
        
        if linear_num >= 2:
            
            linear.append( nn.Linear(in_features=in_out_nz, out_features=nz) )
            linear.append( nn.ELU() )   

            for i in range(linear_num-2) : 
                linear.append( nn.Linear(in_features=nz, out_features=nz) )
                linear.append( nn.ELU() )

            linear.append( nn.Linear(in_features=nz, out_features=in_out_nz) )
            linear.append( nn.ELU() )
        else :
            linear.append( nn.Linear(in_features=in_out_nz, out_features=in_out_nz) )      
                      
        self.linear = linear
        
    def forward(self, input):
        for layer in self.linear : 
            input = layer(input)
        return input

# stack version
class PointMapping(nn.Module) : 
    def __init__(self, nz, linear_num) : 
        super(PointMapping, self).__init__()
        
        linear = nn.ModuleList()
        for i in range(linear_num) : 
            in_features = nz
            out_features = nz
            if i == 0 : in_features = 1
            if i== linear_num-1 : out_features=1
            linear.append( nn.Linear(in_features=in_features, out_features=out_features) )
            if i!=linear_num-1 : linear.append( nn.LeakyReLU(0.1) )
            if i!=linear_num-1 : linear.append( nn.BatchNorm1d(out_features) )
        self.linear = linear
        
    def forward(self, input):
        for layer in self.linear : 
            input = layer(input)
        return input
        
class EachLatentMapping(nn.Module) : 
    def __init__(self, nz, inter_nz, linear_num) :
        super(EachLatentMapping, self).__init__()
        
        pm_list = nn.ModuleList()
        for i in range(nz) : 
            pm = PointMapping(inter_nz, linear_num)
            pm_list.append(pm)
        
        self.pm_list = pm_list
        self.nz = nz
    
    def forward(self, x) : 
        assert x.size(1) == self.nz
        
        predict_out_list = []
        for each_nz in range(self.nz) : 
            each_nz_x = x[:, each_nz].view(-1,1)
            predict = self.pm_list[each_nz](each_nz_x)
            
            predict_out_list.append(predict)
        
        predict_out = torch.cat(predict_out_list, dim=1)
        return predict_out
        
    def point_forward(self, nz_where, x) : 
        each_nz_x = x.view(-1,1)
        predict = self.pm_list[nz_where](each_nz_x)
        return predict
    
    def get_optimizer_list(self, optim_class, **kwargs) : 
        optim_list = []
        #print(**kwargs)
        for i in range(self.nz) : 
            optim_list.append(optim_class(self.pm_list[i].parameters(), **kwargs))
        return optim_list
    
    
    
class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(len(input), self.shape[0], self.shape[1], self.shape[2])
    
    
class Decoder(nn.Module):
    def __init__(self, nz=32, img_size=32, ngpu=1, ndf=64, ngf=64, nc=3):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        
        kernel = 2*img_size//32
        # img_size 32 --> 2
        # img_size 64 --> 4
        # img_size 128 --> 8
        # img_size 256 --> 16
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(     nz, ngf * 8, kernel, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        input = input.view(-1,self.nz, 1, 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



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