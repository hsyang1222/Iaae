from matplotlib import pyplot as plt
import torch
import os
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from PIL import Image

import seaborn as sns
def pca_kde(real, test, encoded, dim=1) : 
    plt.clf()
    
    all_data = torch.cat([real, test, encoded])
    U, S, V = torch.pca_lowrank(all_data)
    
    pca1 = torch.matmul(real, V[:, :dim])
    pca2 = torch.matmul(test, V[:, :dim])
    pca3 = torch.matmul(encoded, V[:, :dim])
    sns.kdeplot(pca1.flatten().numpy(), label='z')
    sns.kdeplot(pca2.flatten().numpy(), label='M(z)')
    plot = sns.kdeplot(pca3.flatten().numpy(), label='E(x)')
    plot.legend()
    return plot

def make_feature_plt(z, M_z, E_x, fnum) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(z, label='z', ax=ax1)
    sns.kdeplot(E_x, label='E(x)', ax=ax1)
    sns.kdeplot(M_z, label='M(z)', ax=ax1)
    ax1.legend()
    ax1.set_title('feature ' + str(fnum))
    return fig1


def feature_plt_list(z, M_z, E_x) : 
    plt.clf()
    plt_list = []
    assert z.size(1) == M_z.size(1) and M_z.size(1) == E_x.size(1)
    for i in range(z.size(1)) :
        plt_list.append(make_feature_plt(z[:,i], M_z[:,i], E_x[:,i], i))
        
    return plt_list

def make_ulearning_dsl(original_train_loader, encoder, device, batch_size):
    encoded_data_list = []
    encoder.eval()
    for each_batch, label in original_train_loader :
        real_image_cuda = each_batch.to(device)
        with torch.no_grad() :            
            encoded_feature = encoder(real_image_cuda).detach().cpu()
            encoded_data_list.append(encoded_feature)
    encoded_feature_tensor = torch.cat(encoded_data_list)

    #각 차원의 feature를 각 차원별로 sort한 tensor가 필요함
    sorted_encoded_feature_tensor = encoded_feature_tensor.clone()
    for each_dim in range(sorted_encoded_feature_tensor.size(1)) : 
        sorted_encoded_feature_tensor[:,each_dim] = torch.sort(encoded_feature_tensor[:,each_dim])[0]

    uniform_input = torch.empty(sorted_encoded_feature_tensor.shape)
    for each_dim in range(sorted_encoded_feature_tensor.size(1)) : 
        uniform_input[:,each_dim] = torch.linspace(0,1,sorted_encoded_feature_tensor.size(0))


    feature_tensor_ds = torch.utils.data.TensorDataset(uniform_input, sorted_encoded_feature_tensor)
    feature_tensor_dloader = torch.utils.data.DataLoader(feature_tensor_ds, batch_size=batch_size, shuffle=True)
                       
    encoder.train()
    return feature_tensor_dloader


def get_encoded_data(train_loader, encoder, device, size=2048) : 
    data, label = next(iter(train_loader))
    assert data.size(0) == size, "not impl"
    data_cuda = data.to(device)
    with torch.no_grad() : 
        encoded_data = encoder(data_cuda)
        return encoded_data.cpu()

def sample_image(encoder, decoder, x):
    z = encoder(x)
    return decoder(z)

def inference_image(mapper, decoder, batch_size, latent_dim, device) :
    # normal distribution
    z = torch.randn(batch_size, latent_dim).to(device)
    return decoder(mapper(z)).detach().cpu()

def inference_image_ulver(mapper, decoder, batch_size, latent_dim, device) :
    # uniform distribution
    z = torch.rand(batch_size, latent_dim).to(device)
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
