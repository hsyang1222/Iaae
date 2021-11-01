from matplotlib import pyplot as plt
import torch
import os
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from PIL import Image
import tqdm
import gc
import time

import seaborn as sns

def timeparse(time_str) : 
    if time_str == '' :
        return -1
    splited = time_str.split(":")
    sec = int(splited[0]) * 3600 + int(splited[1]) * 60 + int(splited[2])
    return sec

def check_time_over(start, limit_sec):
    if limit_sec < 0 :
        return False
    return (time.time() - start) > limit_sec

def save_model(model_list, name_list) : 
    for model,name in zip(model_list, name_list) : 
        if isinstance(model, torch.nn.Module) : 
            torch.save(model.state_dict(), './model/'+name)
        
def load_model(model, name) :
    if model is not None :
        model.load_state_dict(torch.load('./model/'+name))

def insert_sample_image_inception(args, i, epochs, train_loader, mapper, decoder, inception_model_score) : 
    model_name = args.model_name
    latent_dim = args.latent_dim
    device = args.device
    
    for each_batch, label in tqdm.tqdm(train_loader, desc='generate info_image[%d/%d]' % (i, epochs)):    
        with torch.no_grad():
            if model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior'] : 
                sampled_images = inference_image(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device)
            if model_name == 'ulearning' :
                sampled_images = inference_image_ulver(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device)
            if model_name in ['ulearning_point', 'mimic_at_last', 'mimic', 'mimic+non-prior'] :
                sampled_images = inference_image_ulpver(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device)
            if model_name in ['vanilla-mimic'] : 
                sampled_images = inference_image_vanilla_mimic(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device) 
            inception_model_score.put_fake(sampled_images) 
        if args.run_test : break


def gen_matric(wandb, args, train_loader, encoder, mapper, decoder, discriminator, inception_model_score) : 
    model_name = args.model_name
    save_image_interval = args.save_image_interval
    device = args.device

    

    #offload all GAN model to cpu and onload inception model to gpu
    encoder = encoder.eval().to('cpu')
    decoder = decoder.eval().to('cpu')
    if args.mapper_inter_layer > 0 : mapper = mapper.eval().to('cpu')
    if model_name in ['vanilla', 'pointMapping_but_aae', 'mimic+non-prior']: 
        discriminator = discriminator.eval().to('cpu')

    inception_model_score.model_to(device)
    gc.collect()
    torch.cuda.empty_cache()

    #generate fake images info
    inception_model_score.lazy_forward(batch_size=64, device=device, fake_forward=True)
    inception_model_score.calculate_fake_image_statistics()
    metrics, plot = inception_model_score.calculate_generative_score(feature_pca_plot=True)
    metrics.update({'IsNet feature':wandb.Image(plot)})

    #onload all GAN model to gpu and offload inception model to cpu
    inception_model_score.model_to('cpu')
    encoder = encoder.train().to(device)
    decoder = decoder.train().to(device)
    if args.mapper_inter_layer > 0 : mapper = mapper.train().to(device)
    if model_name in ['vanilla', 'pointMapping_but_aae', 'mimic+non-prior'] : 
        discriminator = discriminator.to(device)

    inception_model_score.clear_fake()
    torch.cuda.empty_cache()
        
    return metrics



def wandb_update(wandb, i, args, train_loader, encoder, mapper, decoder, device, fixed_z, loss_log) : 
    latent_dim = args.latent_dim
    model_name = args.model_name
    
    
    fixed_fake_image = get_fixed_z_image_np(args.model_name, decoder, mapper, fixed_z)
    
    real_encoded_data = get_encoded_data(train_loader, encoder, device=device, size=2048)                
    mapper_test_data, mapper_out_data = make_mapper_out(model_name, mapper, real_encoded_data, latent_dim, args.mapper_inter_layer, device) 

    if args.model_name in ['vanilla-mimic']:
        pass
    elif args.mapper_inter_layer :
        show_mapped=True
        show_Ex = True
        show_z = False
        mapper_input_out_plot =  wandb.Image(pca_kde(mapper_test_data, mapper_out_data, \
                                                 real_encoded_data, 1, show_mapped, show_z, show_Ex))
    else : 
        show_mapped=False
        show_Ex = True
        show_z = True
        mapper_input_out_plot =  wandb.Image(pca_kde(mapper_test_data, mapper_out_data, \
                                                 real_encoded_data, 1, show_mapped, show_z, show_Ex))

    if args.model_name in ['vanilla-mimic']:
        z_target = torch.randn(2048, latent_dim)
        M_optimize = mapper(real_encoded_data.to(device), reparm=False).detach().cpu()
        mapper_input_out_plot = wandb.Image(pca_kde_vanilla_mimic(real_encoded_data, M_optimize, z_target))
        feature_kde = feature_plt_list_vanilla_mimic(z_target, M_optimize, real_encoded_data, mapper.mu, mapper.sig)
    elif isinstance(mapper, torch.nn.Module) : 
        feature_kde = feature_plt_list(mapper_test_data, mapper_out_data, real_encoded_data)
    else :
        feature_kde = feature_plt_list(mapper_test_data, mapper_out_data, real_encoded_data, use_M_z=False)

        
        
    loss_log.update({
               "fake_image" :[wandb.Image(fixed_fake_image, caption='fixed z image')],
               "mapper_inout(pca 1dim)" : mapper_input_out_plot,
               "feature_kde" : [wandb.Image(plt) for plt in feature_kde],
              })
  
    wandb.log(loss_log, step=i)

def make_fixed_z(model_name, latent_dim, device):
    if model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior',  'vanilla-mimic'] :
        z = torch.randn(8 ** 2, latent_dim, device=device)
    elif model_name in ['ulearning'] : 
        z = torch.rand(8 ** 2, latent_dim, device=device)
    elif model_name in ['ulearning_point', 'mimic_at_last', 'mimic', 'mimic+non-prior']:
        z = torch.rand(8**2, latent_dim, device=device) * 2 -1 
    return z

def make_mapper_out(model_name, mapper, encoded, latent_dim, mapper_inter_layer, device) : 
    if model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior'] :
        mapper_test_data = torch.randn(2048, latent_dim, device=device)
    elif model_name in ['ulearning'] :
        mapper_test_data = torch.rand(2048,latent_dim, device=device)
    elif model_name in ['ulearning_point', 'mimic_at_last', 'mimic', 'mimic+non-prior']:
        mapper_test_data = torch.rand(2048, latent_dim, device=device) * 2 -1
    elif model_name in ['vanilla-mimic']:
        mapper_test_data = encoded.to(device)
    if mapper_inter_layer > 0 : 
        with torch.no_grad():
            mapper_out_data = mapper(mapper_test_data)
    else : 
        mapper_out_data = mapper_test_data
    return mapper_test_data.cpu(), mapper_out_data.cpu()
    
def pca_kde_vanilla_mimic(encoded_input, mapped_out, z_target, dim=1) : 
    plt.clf()
    #case vanilla-mimic
    all_data = torch.cat([z_target, encoded_input])
    U, S, V = torch.pca_lowrank(all_data)
    z_pca = torch.matmul(z_target, V[:, :dim])
    m_pca = torch.matmul(mapped_out, V[:, :dim])
    e_pca = torch.matmul(encoded_input, V[:, :dim])
    sns.kdeplot(e_pca.flatten().numpy(), label='E(x)-input', alpha=0.6, color='r')
    sns.kdeplot(m_pca.flatten().numpy(), label='M(E(x))-optimize', alpha=0.6, color='b')
    plot = sns.kdeplot(z_pca.flatten().numpy(), label='Z-target', alpha=0.6, color='g')
    plot.legend()
    return plot
    
    
def pca_kde(z_data, m_out_data, encoded, dim=1, show_mapped=False, show_z=False, show_Ex=False) : 
    plt.clf()
    if show_z and not show_mapped: 
        all_data = torch.cat([z_data, encoded])
        U, S, V = torch.pca_lowrank(all_data)
        pca1 = torch.matmul(z_data, V[:, :dim])
        pca3 = torch.matmul(encoded, V[:, :dim])
        sns.kdeplot(pca3.flatten().numpy(), label='E(x)-optimize', alpha=0.6, color='r')
        plot = sns.kdeplot(pca1.flatten().numpy(), label='Z-target', alpha=0.6, color='b')
    if show_mapped and not show_z: 
        all_data = torch.cat([m_out_data, encoded])
        U, S, V = torch.pca_lowrank(all_data)
        pca2 = torch.matmul(m_out_data, V[:, :dim])
        pca3 = torch.matmul(encoded, V[:, :dim])
        sns.kdeplot(pca2.flatten().numpy(), label='M(z)-optimize', alpha=0.6, color='r')
        plot = sns.kdeplot(pca3.flatten().numpy(), label='E(x)-target', alpha=0.6, color='b')
    if show_mapped and not show_Ex :
        #case vanilla-mimic
        all_data = torch.cat([z_data, encoded])
        U, S, V = torch.pca_lowrank(all_data)
        pca1 = torch.matmul(z_data, V[:, :dim])
        pca2 = torch.matmul(m_out_data, V[:, :dim])
        pca3 = torch.matmul(encoded, V[:, :dim])
        sns.kdeplot(pca2.flatten().numpy(), label='M(E(x))-optimize', alpha=0.6, color='b')
        sns.kdeplot(pca3.flatten().numpy(), label='E(x)-input', alpha=0.6, color='r')
        plot = sns.kdeplot(pca1.flatten().numpy(), label='Z-target', alpha=0.6, color='g')
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

def feature_plt_list_vanilla_mimic(z_target, M_optimize, E_x_input, mu, sig) : 
    plt.clf()
    plt_list = []
    assert z_target.size(1) ==M_optimize.size(1) and M_optimize.size(1) == E_x_input.size(1)
    for i in range(z_target.size(1)) :
        #z_target_reparm = z_target[:,i] * sig[i] + mu[i]
        plt_list.append(make_feature_plt_mimic(E_x_input[:,i], M_optimize[:,i], z_target[:,i], i))
    return plt_list

def make_feature_plt_mimic(E_x, M_z, z, fnum) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(z, label='Z-target', ax=ax1, alpha=0.6, color='g')
    sns.kdeplot(E_x, label='E(x)-input', ax=ax1, alpha=0.6, color='r')
    sns.kdeplot(M_z, label='M(E(z))-optimize', ax=ax1, alpha=0.6, color='b')
    ax1.legend()
    ax1.set_title('feature ' + str(fnum))
    return fig1

def make_feature_plt_notMz(z, E_x, fnum) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(z, label='Z-target', ax=ax1, alpha=0.6, color='g')
    sns.kdeplot(E_x, label='E(x)-optimize', ax=ax1, alpha=0.6, color='b')
    ax1.legend()
    ax1.set_title('feature ' + str(fnum))
    return fig1

def feature_plt_list(z, M_z, E_x, use_M_z=True, vanilla_mimic=False) : 
    plt.clf()
    plt_list = []
    assert z.size(1) == M_z.size(1) and M_z.size(1) == E_x.size(1)
    if vanilla_mimic : 
        for i in range(z.size(1)) :
            plt_list.append(make_feature_plt_mimic(z[:,i], M_z[:,i], E_x[:,i], i))
    elif use_M_z : 
        for i in range(z.size(1)) :
            plt_list.append(make_feature_plt(z[:,i], M_z[:,i], E_x[:,i], i))
    else:
        for i in range(z.size(1)) :
            plt_list.append(make_feature_plt_notMz(z[:,i], E_x[:,i], i))
    
    return plt_list

def make_encoded_feature_tensor(encoder, train_loader, device):
    encoded_data_list = []
    for each_batch, label in tqdm.tqdm(train_loader, desc='make M featrue') :
        real_image_cuda = each_batch.to(device)
        with torch.no_grad() :            
            encoded_feature = encoder(real_image_cuda).detach().cpu()
            encoded_data_list.append(encoded_feature)
    encoded_feature_tensor = torch.cat(encoded_data_list)
    return encoded_feature_tensor

def make_sorted_encoded_feature_tensor(encoded_feature_tensor) : 
    sorted_encoded_data = encoded_feature_tensor[:,0].sort()[0].view(-1,1)
    return sorted_encoded_data

def make_linspace_tensor(data):
    linspace_list = []
    for i in range(data.size(1)) : 
        linspace_list.append(torch.linspace(-1,1,data.size(0)).view(-1,1))
    linspace = torch.cat(linspace_list,dim=1)
    return linspace

def encoded_feature_to_dl(encoded_feature_tensor, batch_size) : 
    #각 차원의 feature를 각 차원별로 sort한 tensor가 필요함
    sorted_encoded_feature_tensor = torch.empty(encoded_feature_tensor.shape)
    for each_dim in range(sorted_encoded_feature_tensor.size(1)) : 
        sorted_encoded_feature_tensor[:,each_dim] = torch.sort(encoded_feature_tensor[:,each_dim])[0]

    # linspace로 입력할 -1~1사이의 균일한 값이 필요함    
    uniform_input = torch.empty(sorted_encoded_feature_tensor.shape)
    for each_dim in range(sorted_encoded_feature_tensor.size(1)) : 
        uniform_input[:,each_dim] = torch.linspace(-1,1,sorted_encoded_feature_tensor.size(0))


    feature_tensor_ds = torch.utils.data.TensorDataset(uniform_input, sorted_encoded_feature_tensor)
    feature_tensor_dloader = torch.utils.data.DataLoader(feature_tensor_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12)
    return feature_tensor_dloader

from scipy.stats import norm
import matplotlib.pyplot as plt
def encoded_feature_to_Ex_z(encoded_feature_tensor, batch_size) : 
    #각 차원의 feature를 각 차원별로 sort한 tensor가 필요함
    sorted_encoded_feature_tensor = torch.empty(encoded_feature_tensor.shape)
    mu = torch.zeros(sorted_encoded_feature_tensor.size(1))
    sig = torch.ones(sorted_encoded_feature_tensor.size(1))
    for each_dim in range(sorted_encoded_feature_tensor.size(1)) : 
        sorted_encoded_feature_tensor[:,each_dim] = torch.sort(encoded_feature_tensor[:,each_dim])[0]
        mu[each_dim] = torch.mean(sorted_encoded_feature_tensor[:,each_dim])
        sig[each_dim] = torch.var(sorted_encoded_feature_tensor[:,each_dim]) ** 0.5

    # gaussian을 따르는 value를 만든다음에 sort하여 사용
    uniform_input = torch.empty(sorted_encoded_feature_tensor.shape)
    for each_dim in range(sorted_encoded_feature_tensor.size(1)) : 
        x = np.linspace(1e-12, 1- 1e-12, sorted_encoded_feature_tensor.size(0))
        data = norm.ppf(x) * sig[each_dim].numpy() + mu[each_dim].numpy()
        uniform_input[:,each_dim] = torch.tensor(data, dtype=torch.float64)

    # Ex를 -1~1사이의 균일한 값으로 mapping하는 dataloader!
    feature_tensor_ds = torch.utils.data.TensorDataset(sorted_encoded_feature_tensor, uniform_input)
    feature_tensor_dloader = torch.utils.data.DataLoader(feature_tensor_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12)
    return feature_tensor_dloader, mu, sig
    
    

def make_ulearning_dsl(original_train_loader, encoder, device, batch_size):
    encoded_data_list = []
    encoder.eval()
    for each_batch, label in original_train_loader :
        real_image_cuda = each_batch.to(device)
        with torch.no_grad() :            
            encoded_feature = encoder(real_image_cuda).detach().cpu()
            encoded_data_list.append(encoded_feature)
    encoded_feature_tensor = torch.cat(encoded_data_list)

    feature_tensor_dloader = encoded_feature_to_dl(encoded_feature_tensor, batch_size)
    
    encoder.train()
    return feature_tensor_dloader

def get_encoded_data(train_loader, encoder, device, size=2048) :
    encoded_data_list = []
    num_data = 0
    for data, label in train_loader : 
        data_cuda = data.to(device)
        encoded_data = encoder(data_cuda)
        encoded_data_list.append(encoded_data.detach().cpu())
        
        num_data += train_loader.batch_size
        if num_data > size : 
            return torch.cat(encoded_data_list)[:size]

def sample_image(encoder, decoder, x):
    z = encoder(x)
    return decoder(z)

def inference_image_lin(decoder, batch_size, latent_dim, device) :
    with torch.no_grad() : 
        z = torch.rand(batch_size, latent_dim).to(device) * 2 - 1
        result = decoder(z).detach().cpu()
    return result

def inference_image_vanilla_mimic(mapper, decoder, batch_size, latent_dim, device) :
    # normal distribution
    with torch.no_grad() : 
        z = torch.randn(batch_size, latent_dim).to(device)
        for each_dim in range(len(mapper.mu)) : 
            z[:,each_dim] = z[:,each_dim] * mapper.sig[each_dim] + mapper.mu[each_dim]
        result = decoder(z).detach().cpu()
    return result


def inference_image(mapper, decoder, batch_size, latent_dim, device) :
    # normal distribution
    with torch.no_grad() : 
        z = torch.randn(batch_size, latent_dim).to(device)
        #z = torch.sigmoid(z)
        result = decoder(mapper(z)).detach().cpu()
    return result

def inference_image_ulver(mapper, decoder, batch_size, latent_dim, device) :
    # uniform distribution
    with torch.no_grad():    
        z = torch.rand(batch_size, latent_dim).to(device)
        result = decoder(mapper(z)).detach().cpu()
    return result

def inference_image_ulpver(mapper, decoder, batch_size, latent_dim, device) :
    # uniform distribution
    with torch.no_grad():
        z = torch.rand(batch_size, latent_dim).to(device) * 2 -1
        result=decoder(mapper(z)).detach().cpu()
    return result 


import torchvision.utils as vutils
def get_fixed_z_image_np(model_name, decoder, mapper, fixed_z, nrow=8) : 
    if model_name in ['vanilla-mimic'] : 
        z = fixed_z.clone()
        for each_dim in range(len(mapper.mu)) : 
            z[:,each_dim] = z[:,each_dim] * mapper.sig[each_dim] + mapper.mu[each_dim]
        fake_image = decoder(z)
    else : 
        fake_image = decoder(mapper(fixed_z))
    
    fake_image = fake_image * 0.5 + 0.5
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
