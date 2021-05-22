import itertools
import numpy as np
import torch
import torch.nn as nn
import generative_model_score
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import os
import wandb
import argparse
import tqdm
from dataset import *
from train import *
from models import *
from utils import *
import hashlib

def main(args):
    # load real images info or generate real images info
    model_name = args.model_name
    torch.cuda.set_device(device=args.device)
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    save_image_interval = args.save_image_interval
    loss_calculation_interval = args.loss_calculation_interval
    latent_dim = args.latent_dim
    project_name = args.project_name
    dataset = args.dataset
    lr = args.lr
    n_iter = args.n_iter

    image_shape = [3, img_size, img_size]

    '''
    customize
    '''
    if model_name in ['vanilla', 'yeop_n_iter', 'yeop_loss']:
        encoder = Encoder(latent_dim, image_shape).cuda()
        decoder = Decoder(latent_dim, image_shape).cuda()
        discriminator = Discriminator(latent_dim).cuda()
    else:
        raise Exception('model name is wrong')

    ###########################################
    #####              Score              #####
    ###########################################
    inception_model_score = generative_model_score.GenerativeModelScore()
    inception_model_score.lazy_mode(True)
    

    '''
    dataset 채워주세요!
    customize
    '''
    if dataset == 'CelebA':
        train_loader, validation_loader, test_loader = get_celebA_dataset(batch_size, img_size)
    elif dataset == 'FFHQ':
        train_loader, test_loader = get_ffhq_thumbnails(batch_size, image_size)
    elif dataset == 'LSUN_bedroom':
        pass
    elif dataset == 'LSUN_cars':
        pass
    elif dataset == 'LSUN_cats':
        pass
    elif dataset == 'LSUN_churches':
        pass
    else:
        train_loader = get_cifar1_dataset(batch_size, img_size)
    
    
    real_images_info_file_name = hashlib.md5(str(train_loader.dataset).encode()).hexdigest()+'.pickle'
    
    os.makedirs('inception_model_info', exist_ok=True)
    if os.path.exists('./inception_model_info/' + real_images_info_file_name) : 
        print("Using generated real image info.")
        print(train_loader.dataset)
        inception_model_score.load_real_images_info('./inception_model_info/' + real_images_info_file_name)

    else : 
        inception_model_score.model_to('cuda')
        
        #put real image
        for each_batch in train_loader : 
            X_train_batch = each_batch[0]
            inception_model_score.put_real(X_train_batch)

        #generate real images info
        inception_model_score.lazy_forward(batch_size=batch_size, device='cuda', real_forward=True)
        inception_model_score.calculate_real_image_statistics()
        #save real images info for next experiments
        inception_model_score.save_real_images_info('./inception_model_info/' + real_images_info_file_name)
        #offload inception_model
        inception_model_score.model_to('cpu')
    

    '''
    customize
    '''
    ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    r_losses = []
    d_losses = []
    g_losses = []
    precisions = []
    recalls = []
    fids = []
    inception_scores_real = []
    inception_scores_fake = []
    

    wandb.login()
    wandb.init(project=project_name, 
               config={
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "img_size": img_size,
                        "save_image_interval": save_image_interval,
                        "loss_calculation_interval": loss_calculation_interval,
                        "latent_dim": latent_dim,
                        "lr": lr,
                        "dataset": dataset,
                        "model_name": model_name,
                        "n_iter": n_iter
                    })
    config = wandb.config


    for i in range(0, epochs):
        batch_count = 0

        for each_batch in tqdm.tqdm(train_loader):
            batch_count += 1
            X_train_batch = Variable(each_batch[0]).cuda()

            '''
            customize
            '''
            r_loss = update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder)
            
            if model_name in ['vanilla']:
                d_loss = update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)

            elif model_name == 'yeop_loss':
                d_loss = update_discriminator_add_loss(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)
            
            elif model_name == 'yeop_n_iter':
                for iter_ in range(n_iter):
                    d_loss = update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)
            else:
                raise Exception('model name is wrong')
            
            g_loss = update_generator(g_optimizer, X_train_batch, encoder, discriminator)

            sampled_images = sample_image(encoder, decoder, X_train_batch).detach().cpu()

            if i % loss_calculation_interval == 0:
                inception_model_score.put_fake(sampled_images)
        
        if i % save_image_interval == 0:
            image = save_images(n_row=10, epoch=i, latent_dim=latent_dim, 
                            model=decoder, dataset=dataset, model_name=model_name)
            wandb.log({'image':wandb.Image(image, caption='%s_epochs' % i)}, step=i)

        if i % loss_calculation_interval == 0:
            #offload all GAN model to cpu and onload inception model to gpu
            encoder = encoder.to('cpu')
            decoder = decoder.to('cpu')
            discriminator = discriminator.to('cpu')
            inception_model_score.model_to('cuda')
            
            #generate fake images info
            inception_model_score.lazy_forward(batch_size=64, device='cuda', fake_forward=True)
            inception_model_score.calculate_fake_image_statistics()
            metrics = inception_model_score.calculate_generative_score()
            
            #onload all GAN model to gpu and offload inception model to cpu
            inception_model_score.model_to('cpu')
            encoder = encoder.to('cuda')
            decoder = decoder.to('cuda')
            discriminator = discriminator.to('cuda')
            
            precision, recall, fid, inception_score_real, inception_score_fake, density, coverage = \
                metrics['precision'], metrics['recall'], metrics['fid'], metrics['real_is'], metrics['fake_is'], metrics['density'], metrics['coverage']
            
            wandb.log({"precision": precision, 
                       "recall": recall,
                       "fid": fid,
                       "inception_score_real": inception_score_real,
                       "inception_score_fake": inception_score_fake,
                       "density": density,
                       "coverage": coverage}, 
                      step=i)
            
            r_losses.append(r_loss)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            precisions.append(precision)
            recalls.append(recall)
            fids.append(fid)
            inception_scores_real.append(inception_score_real)
            inception_scores_fake.append(inception_score_fake)
            save_scores_and_print(i + 1, epochs, r_loss, d_loss, g_loss, precision, recall, fid, 
                                  inception_score_real, inception_score_fake, dataset, model_name)
            
        inception_model_score.clear_fake()
    save_losses(epochs, loss_calculation_interval, r_losses, d_losses, g_losses)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--save_image_interval', type=int, default=5)
    parser.add_argument('--loss_calculation_interval', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--project_name', type=str, default='AAE')
    parser.add_argument('--dataset', type=str, default='', choices=['LSUN_churches', 'LSUN_bedroom',
                                                                    'LSUN_cars', 'LSUN_cats',
                                                                    'FFHQ', 'CelebA', 'cifar10'])

    parser.add_argument('--model_name', type=str, default='', choices=['vanilla', 'yeop_loss', 
                                                                       'yeop_n_iter', ''])

    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)