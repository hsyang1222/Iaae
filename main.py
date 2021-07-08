import generative_model_score
inception_model_score = generative_model_score.GenerativeModelScore()
import itertools
import numpy as np
import torch
import torch.nn as nn
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
    global inception_model_score
    
    # load real images info or generate real images info
    model_name = args.model_name
    #torch.cuda.set_device(device=args.device)
    device = args.device
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
    
    wandb.login()
    wandb_name = dataset+','+model_name +','+str(img_size)+",infr_sample"
    if args.run_test : wandb_name += ', test run'
    wandb.init(project=project_name, 
               config=args,
               name = wandb_name)
    config = wandb.config

    '''
    customize
    '''
    if model_name in ['vanilla', 'yeop_n_iter', 'yeop_loss']:
        encoder = Encoder(latent_dim, image_shape).to(device)
        decoder = Decoder(latent_dim, image_shape).to(device)
        discriminator = Discriminator(latent_dim).to(device)
    elif model_name == "mod_var" :
        encoder = ModEncoder(latent_dim, image_shape).to(device)
        decoder = ModDecoder(latent_dim, image_shape).to(device)
        discriminator = Discriminator(latent_dim).to(device)
    elif model_name == "mod2_var" :
        encoder = Mod2Encoder(latent_dim, image_shape).to(device)
        decoder = Mod2Decoder(latent_dim, image_shape).to(device)
        discriminator = Discriminator(latent_dim).to(device)
    elif model_name == 'gme' : 
        encoder = GME_Encoder(latent_dim, image_shape).to(device)
        decoder = GME_Decoder(latent_dim, image_shape).to(device)
        discriminator = GME_Discriminator(image_shape).to(device)
    elif model_name == "latent_mapping" :
        encoder = Encoder(latent_dim, image_shape).to(device)
        decoder = StackDecoder(latent_dim, image_shape).to(device)
        discriminator = Discriminator(latent_dim).to(device)
    elif model_name == "direct" :
        encoder = DirectEncoder(latent_dim, image_shape).to(device)
        decoder = Decoder(latent_dim, image_shape).to(device)
        z_dis = Discriminator(latent_dim).to(device)
        img_dis = ImageDiscriminator(image_shape).to(device)
    else:
        raise Exception('model name is wrong')

    ###########################################
    #####              Score              #####
    ###########################################
    inception_model_score.lazy_mode(True)
    

    '''
    dataset 채워주세요!
    customize
    '''
    if dataset == 'CelebA':
        train_loader = get_celebA_dataset(batch_size, img_size)
    elif dataset == 'FFHQ':
        train_loader, test_loader = get_ffhq_thumbnails(batch_size, img_size)
    elif dataset == 'mnist':
        train_loader = get_mnist_dataset(batch_size, img_size)
    elif dataset == 'mnist_fashion':
        train_loader = get_mnist_fashion_dataset(batch_size, img_size)
    elif dataset == 'emnist':
        train_loader = get_emnist_dataset(batch_size, img_size)
    elif dataset == 'LSUN_dining_room':
        #wget http://dl.yf.io/lsun/scenes/dining_room_train_lmdb.zip
        #unzip dining_room_train_lmdb.zip
        #located dining_room_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='dining_room_train')
    elif dataset == 'LSUN_classroom':
        #wget http://dl.yf.io/lsun/scenes/classroom_train_lmdb.zip
        #unzip classroom_train_lmdb.zip
        #located classroom_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='classroom_train')
    elif dataset == 'LSUN_conference':
        #wget http://dl.yf.io/lsun/scenes/conference_room_train_lmdb.zip
        #unzip conference_room_train_lmdb.zip
        #located conference_room_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='conference_room_train')
    elif dataset == 'LSUN_churches':
        #wget http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip
        #unzip church_outdoor_train_lmdb.zip
        #located church_outdoor_train_lmdb folder in dataset directory
        train_loader = get_lsun_dataset(batch_size, img_size, classes='church_outdoor_train')
    else:
        print("dataset is forced selected to cifar10")
        train_loader = get_cifar1_dataset(batch_size, img_size)
    
    
    real_images_info_file_name = hashlib.md5(str(train_loader.dataset).encode()).hexdigest()+'.pickle'
    if args.run_test : real_images_info_file_name += '.run_test' 
    
    os.makedirs('../../inception_model_info', exist_ok=True)
    if os.path.exists('../../inception_model_info/' + real_images_info_file_name) : 
        print("Using generated real image info.")
        print(train_loader.dataset)
        inception_model_score.load_real_images_info('../../inception_model_info/' + real_images_info_file_name)

    else : 
        inception_model_score.model_to(device)
        
        #put real image
        for each_batch in tqdm.tqdm(train_loader, desc='insert real dataset') : 
            X_train_batch = each_batch[0]
            inception_model_score.put_real(X_train_batch)
            if args.run_test : break

        #generate real images info
        inception_model_score.lazy_forward(batch_size=64, device=device, real_forward=True)
        inception_model_score.calculate_real_image_statistics()
        #save real images info for next experiments
        inception_model_score.save_real_images_info('../../inception_model_info/' + real_images_info_file_name)
        #offload inception_model
        inception_model_score.model_to('cpu')
    

    '''
    customize
    '''
    
    if model_name == 'direct' : 
        e_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        g_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        zd_optimizer = torch.optim.Adam(z_dis.parameters(), lr=lr)
        imgd_optimizer = torch.optim.Adam(img_dis.parameters(), lr=lr)
        bce = torch.nn.BCELoss()
        mse = torch.nn.MSELoss()
    else : 
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
    


    for i in range(0, epochs):
        batch_count = 0

        for each_batch in tqdm.tqdm(train_loader, desc='train batch'):
            batch_count += 1
            X_train_batch = Variable(each_batch[0]).to(device)
            
            
            if model_name == 'direct' :
                batch_size = X_train_batch.size(0)
                true_label = torch.ones((batch_size,1), device=device)
                false_label = torch.zeros((batch_size,1), device=device)
                #
                #make distribution of encode(x) to Gaussian
                #
                
                #optim z_dis
                ##z_dis(gaussian) --> True               
                gaussian_sample = torch.randn((batch_size, latent_dim), device=device)
                zd_say_gaussian_is = z_dis(gaussian_sample.detach())
                zd_gaussian_true = bce(zd_say_gaussian_is, true_label)
       
                ##z_dis(E(x)) --> False                
                real_z = encoder(X_train_batch)
                zd_say_realz_is = z_dis(real_z)
                zd_realz_false = bce(zd_say_realz_is, false_label)
                
                zd_optimizer.zero_grad()
                zd_loss = zd_gaussian_true + zd_realz_false
                zd_loss.backward()
                zd_optimizer.step()
                
                #optim e
                ##z_dis(E(x)) --> True
                real_z = encoder(X_train_batch)
                zd_say_realz_is = z_dis(real_z)
                zd_realz_true = bce(zd_say_realz_is, true_label)
                
                e_optimizer.zero_grad()
                zd_realz_true.backward()
                e_optimizer.step()
                
                #
                #make distribution of G(z) to real data
                #
                
                #optim img_dis
                ##img_dis(real_img) --> True
                imgd_say_realimg_is = img_dis(X_train_batch)
                imgd_realimg_true = bce(imgd_say_realimg_is, true_label)
                
                ##img_dis(G(z)) --> False
                fake_img = decoder(gaussian_sample.detach())
                imgd_say_fakeimg_is = img_dis(fake_img)
                imgd_fakeimg_false = bce(imgd_say_fakeimg_is, false_label)
                
                imgd_optimizer.zero_grad()
                imgd_loss = imgd_realimg_true + imgd_fakeimg_false
                imgd_loss.backward()
                
                #optim G
                ##img_dis(G(z)) --> True
                fake_img = decoder(gaussian_sample.detach())
                imgd_say_fakeimg_is = img_dis(fake_img)
                imgd_fakeimg_true = bce(imgd_say_fakeimg_is, true_label)
                
                g_optimizer.zero_grad()
                imgd_fakeimg_true.backward()
                g_optimizer.step()
                
                #
                # make G(E(x)) --> x only if z_dis(E(x))=True and img_dis(G(E(x))=True
                #
            
                #condition
                real_z = encoder(X_train_batch)
                repaint_x = decoder(real_z)
                condition_z = z_dis(real_z).detach() >= 0.5
                condition_img = img_dis(repaint_x).detach() >= 0.5
                condition_onlyif = (condition_z & condition_img)
                percent_onlyif = torch.sum(condition_onlyif) / len(condition_onlyif)
                
                #optim E and G
                repaint_loss = mse(repaint_x, X_train_batch) * percent_onlyif
                e_optimizer.zero_grad()
                g_optimizer.zero_grad()
                repaint_loss.backward()
                e_optimizer.step()
                g_optimizer.step()
                
                
                r_loss = repaint_loss
                d_loss = imgd_loss + zd_loss
                g_loss = imgd_fakeimg_true
                
            else :     

                '''
                customize
                '''
                if model_name == 'gme' : 
                    r_loss = torch.zeros(1)
                else :
                    r_loss = update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder)

                if model_name == 'yeop_loss':
                    d_loss = update_discriminator_add_loss(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)
                elif model_name == 'yeop_n_iter':
                    for iter_ in range(n_iter):
                        d_loss = update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)
                elif model_name == 'gme':
                    d_loss = gme_update_discriminator(d_optimizer, X_train_batch, encoder, decoder, discriminator, latent_dim)
                else:
                    d_loss = update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim)

                if model_name == 'gme' :
                     g_loss = gme_update_generator(g_optimizer, X_train_batch, encoder, decoder, discriminator)
                else:
                     g_loss = update_generator(g_optimizer, X_train_batch, encoder, discriminator)

                
            sampled_images = inference_image(decoder, batch_size=X_train_batch.size(0), latent_dim=latent_dim, device=device).detach().cpu()
            if i % loss_calculation_interval == 0:
                if model_name == 'gme_inference' : 
                    pass
                else : 
                    inception_model_score.put_fake(sampled_images)

            if args.run_test : break
        
        if i % save_image_interval == 0:
            image = save_images(n_row=10, epoch=i, latent_dim=latent_dim, 
                            model=decoder, dataset=dataset, model_name=model_name, device=device)
            wandb.log({'image':wandb.Image(image, caption='%s_epochs' % i)}, step=i)

        if i % loss_calculation_interval == 0:
            #offload all GAN model to cpu and onload inception model to gpu
            encoder = encoder.to('cpu')
            decoder = decoder.to('cpu')
            if model_name == 'direct' : 
                z_dis = z_dis.to('cpu')
                img_dis = img_dis.to('cpu')
            else : 
                discriminator = discriminator.to('cpu')
            inception_model_score.model_to(device)
            
            #generate fake images info
            inception_model_score.lazy_forward(batch_size=64, device=device, fake_forward=True)
            inception_model_score.calculate_fake_image_statistics()
            metrics = inception_model_score.calculate_generative_score()
            
            #onload all GAN model to gpu and offload inception model to cpu
            inception_model_score.model_to('cpu')
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            if model_name == 'direct' : 
                z_dis = z_dis.to(device)
                img_dis = img_dis.to(device)
            else : 
                discriminator = discriminator.to(device)
            
            precision, recall, fid, inception_score_real, inception_score_fake, density, coverage = \
                metrics['precision'], metrics['recall'], metrics['fid'], metrics['real_is'], metrics['fake_is'], metrics['density'], metrics['coverage']
            
            if model_name == 'mod_var':
                wandb.log({"precision": precision, 
                           "recall": recall,
                           "fid": fid,
                           "inception_score_real": inception_score_real,
                           "inception_score_fake": inception_score_fake,
                           "density": density,
                           "coverage": coverage, 
                           'encoder V' : torch.sigmoid(encoder.model[3].v),
                           'decoder V' : torch.sigmoid(decoder.model[3].v),
                          }, step=i)
            elif model_name == 'direct' : 
                wandb.log({"precision": precision, 
                           "recall": recall,
                           "fid": fid,
                           "inception_score_real": inception_score_real,
                           "inception_score_fake": inception_score_fake,
                           "density": density,
                           "coverage": coverage, 
                           "zd_gaussian_true" : zd_gaussian_true,
                           "zd_realz_false" :zd_realz_false,
                           "zd_realz_true" :zd_realz_true,
                           "imgd_realimg_true" :imgd_realimg_true,
                           "imgd_fakeimg_false" :imgd_fakeimg_false,
                           "imgd_fakeimg_true" :imgd_fakeimg_true,
                           "percent_onlyif" :percent_onlyif,
                           "repaint_loss" :repaint_loss
                          }, step=i)
            else : 
                wandb.log({"precision": precision, 
                           "recall": recall,
                           "fid": fid,
                           "inception_score_real": inception_score_real,
                           "inception_score_fake": inception_score_fake,
                           "density": density,
                           "coverage": coverage, 
                          }, step=i)
            
            r_losses.append(r_loss.item())
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            precisions.append(precision)
            recalls.append(recall)
            fids.append(fid)
            inception_scores_real.append(inception_score_real)
            inception_scores_fake.append(inception_score_fake)
            save_scores_and_print(i + 1, epochs, r_loss, d_loss, g_loss, precision, recall, fid, 
                                  inception_score_real, inception_score_fake, dataset, model_name)
            
        inception_model_score.clear_fake()
    #save_losses(epochs, loss_calculation_interval, r_losses, d_losses, g_losses)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--save_image_interval', type=int, default=5)
    parser.add_argument('--loss_calculation_interval', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--project_name', type=str, default='AAE_exact')
    parser.add_argument('--dataset', type=str, default='', choices=['LSUN_dining_room', 'LSUN_classroom', 'LSUN_conference', 'LSUN_churches',
                                                                    'FFHQ', 'CelebA', 'cifar10', 'mnist', 'mnist_fashion', 'emnist'])

    parser.add_argument('--model_name', type=str, default='', choices=['vanilla', 'yeop_loss', 
                                                                       'yeop_n_iter', 'mod_var', 'mod2_var', 'gme', 'latent_mapping', 'gme_inference', 'direct'])

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run_test', type=bool, default=False)

    args = parser.parse_args()

    main(args)
