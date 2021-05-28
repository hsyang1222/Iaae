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
    z_prior = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).cuda()
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

    total_d_loss = 0.1 * kld + d_loss
    total_d_loss.backward()
    d_optimizer.step()
    return total_d_loss.data


# 한누리
def update_autoencoder_with_feature(ae_optimizer, X_train_batch, encoder, decoder, inception_model_score, each_epoch,
                                    batch_count):
    ae_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    X_decoded = decoder(z_posterior)
    l1_loss = torch.nn.L1Loss()
    X_train_batch_feature = inception_model_score.get_hidden_representation(X_train_batch.cpu(), real=True,
                                                                            epoch=each_epoch, batch=batch_count)
    X_decoded_feature = inception_model_score.get_hidden_representation(X_decoded.cpu())
    featurewise_loss = l1_loss(X_decoded_feature, X_train_batch_feature)
    pixelwise_loss = l1_loss(X_decoded, X_train_batch)
    r_loss = 0.9 * pixelwise_loss + 0.1 * featurewise_loss
    r_loss.backward()
    ae_optimizer.step()
    return r_loss
