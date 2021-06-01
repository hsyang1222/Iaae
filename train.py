import torch
from torch.autograd import Variable
import numpy as np

def update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder):
    ae_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    X_decoded = decoder(z_posterior)
    pixelwise_loss = torch.nn.L1Loss()
    r_loss = pixelwise_loss(X_decoded, X_train_batch)
    r_loss.backward()
    ae_optimizer.step()
    return r_loss


def update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim):
    d_optimizer.zero_grad()
    batch_size = X_train_batch.size(0)
    z_prior = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(X_train_batch.device)
    z_posterior = encoder(X_train_batch)
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(z_posterior)))
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


def update_generator(g_optimizer, X_train_batch, encoder, discriminator):
    g_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    g_loss = -torch.mean(torch.log(discriminator(z_posterior)))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data


#################################################################################################
# 양현식

def gme_update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder):
    ae_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    X_decoded = decoder(z_posterior)
    pixelwise_loss = torch.nn.L1Loss()
    r_loss = pixelwise_loss(X_decoded, X_train_batch)
    r_loss.backward()
    ae_optimizer.step()
    return r_loss


def gme_update_discriminator(d_optimizer, X_train_batch, encoder, decoder, discriminator, latent_dim):
    bce = torch.nn.BCELoss()
    d_optimizer.zero_grad()
    batch_size = X_train_batch.size(0)
    device=X_train_batch.device
    
    real_label = torch.ones(batch_size, device=device)
    predict_about_real = discriminator(X_train_batch)
    loss_about_real = bce(predict_about_real, real_label)
    
    fake_label = torch.zeros(batch_size, device=device)
    encoded_z = encoder(X_train_batch)
    fake = decoder(encoded_z)
    predict_about_fake = discriminator(fake)
    loss_about_fake = bce(predict_about_fake, fake_label)
    
    d_loss = loss_about_real + loss_about_fake
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


def gme_update_generator(g_optimizer, X_train_batch, encoder, decoder, discriminator):
    bce = torch.nn.BCELoss()
    g_optimizer.zero_grad()
    batch_size = X_train_batch.size(0)
    device=X_train_batch.device
    
    real_label = torch.ones(batch_size, device=device)
    
    encoded_z = encoder(X_train_batch)
    fake = decoder(encoded_z)
    predict_about_fake = discriminator(fake)
    loss_about_fake = bce(predict_about_fake, real_label)

    g_loss = loss_about_fake
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data

# 김상엽
def update_discriminator_add_loss(d_optimizer, X_train_batch, encoder, discriminator, latent_dim):
    d_optimizer.zero_grad()
    batch_size = X_train_batch.size(0)
    from torch.autograd import Variable
    z_prior = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).cuda()
    z_posterior = encoder(X_train_batch)
    
    kld_criterion = nn.KLDivLoss()
    kld = kld_criterion(F.log_softmax(z_posterior), z_prior)
    
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(z_posterior)))
    
    total_d_loss = 0.1*kld + d_loss
    total_d_loss.backward()
    d_optimizer.step()
    return total_d_loss.data

# 한누리