from matplotlib import pyplot as plt
import torch
import os
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from PIL import Image

import seaborn as sns
def pca_kde(real, test, dim=1) : 
    plt.clf()
    
    real_fake = torch.cat([real, test])
    U, S, V = torch.pca_lowrank(real_fake)
    
    pca1 = torch.matmul(real, V[:, :dim])
    pca2 = torch.matmul(test, V[:, :dim])
    sns.kdeplot(pca1.flatten().numpy(), label='mapper input')
    plot = sns.kdeplot(pca2.flatten().numpy(), label='mapper output')
    plot.legend()
    return plot

def sample_image(encoder, decoder, x):
    z = encoder(x)
    return decoder(z)

def inference_image(mapper, decoder, batch_size, latent_dim, device) :
    z = torch.randn(batch_size, latent_dim).to(device)
    return decoder(mapper(z)).detach().cpu()

import torchvision.utils as vutils
def get_fixed_z_image_np(fake_image, nrow=8) : 
    fake_np = vutils.make_grid(fake_image.detach().cpu(), nrow).permute(1,2,0).numpy()
    return fake_np  

def save_losses(epochs, save_calculation_interval, r_losses, d_losses, g_losses):
    X = range(1, epochs + 1, save_calculation_interval)
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(3, 1, 1)
    plt.title("r_losses")
    plt.plot(X, r_losses, color="blue", linestyle="-", label="r_losses")
    plt.subplot(3, 1, 2)
    plt.title("g_losses")
    plt.plot(X, g_losses, color="purple", linestyle="-", label="g_losses")
    plt.subplot(3, 1, 3)
    plt.title("d_losses")
    plt.plot(X, d_losses, color="red", linestyle="-", label="d_losses")
    plt.savefig('aae_celebA/losses.png')
    plt.close()

def save_scores_and_print(current_epoch, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real,
                          inception_score_fake, dataset, model_name):
    folder_name = 'logs/%s_%s' % (dataset, model_name)
    os.makedirs(folder_name, exist_ok=True)
    f = open("./%s/generative_scores.txt" % folder_name, "a")
    f.write("%d %f %f %f %f %f %f %f %f\n" % (current_epoch, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake))
    f.close()
    print("[Epoch %d/%d] [R loss: %f] [D loss: %f] [G loss: %f] [precision: %f] [recall: %f] [fid: %f] [inception_score_real: %f] [inception_score_fake: %f]"
          % (current_epoch, epochs, r_loss, d_loss, g_loss, precision, recall, fid, inception_score_real, inception_score_fake))


def save_images(n_row, epoch, latent_dim, model, dataset, model_name, device):
    folder_name = '%s_%s' % (dataset, model_name)
    os.makedirs('images/%s' % folder_name, exist_ok=True)
    """Saves a grid of generated digits"""
    # Sample noise
    z = torch.tensor(  np.random.normal(0, 1,(n_row ** 2, latent_dim) )).float().to(device)
    gen_imgs = model(z)
    image_name = "images/%s/%d_epoch.png" % (folder_name, epoch)
    save_image(gen_imgs.data, image_name, nrow=n_row, normalize=True)
    return Image.open(image_name)
