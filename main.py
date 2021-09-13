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
import matplotlib.pyplot as plt

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
    latent_layer = args.latent_layer
    
    fixed_z = make_fixed_z(model_name, latent_dim, device)

    image_shape = [3, img_size, img_size]
    

    if args.wandb : 
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
    if model_name in ['vanilla', 'pointMapping_but_aae']:
        encoder = Encoder(latent_dim, image_shape).to(device)
        decoder = Decoder(latent_dim, image_shape).to(device)
        discriminator = Discriminator(latent_dim).to(device)
        ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        
    elif model_name in ['ulearning', 'ulearning_point'] : 
        encoder = Encoder(latent_dim, image_shape).to(device)
        decoder = Decoder(latent_dim, image_shape).to(device)
        ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr)

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
   
    if args.latent_layer > 0 : 
        if model_name in ['ulearning_point']:
            mapper = EachLatentMapping(nz=32, inter_nz=args.mapper_inter_nz, linear_num=args.mapper_inter_layer).to(device)
        elif model_name in [ 'pointMapping_but_aae']:
            mapper = EachLatentMapping(nz=32, inter_nz=args.mapper_inter_nz, linear_num=args.mapper_inter_layer).to(device)
            m_optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
        elif model_name in ['ulearning', 'vanilla'] : 
            mapper = Mapping(latent_dim, args.latent_layer).to(device)
            m_optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    else :
        mapper = lambda x : x

    for i in range(0, args.AE_iter) :
        for each_batch, label in tqdm.tqdm(train_loader, desc='train AE[%d/%d]' % (i, args.AE_iter)) :
            real_image = each_batch.to(device)
            loss_r = update_autoencoder(ae_optimizer, real_image, encoder, decoder)
            if args.run_test : break
        #print(i, loss_r.item())
        if args.run_test : break
    
    #pretrain M layer
    if args.latent_layer > 0 :
        encoder.eval()
        if model_name in ['ulearning'] : 

            feature_tensor_dloader = make_ulearning_dsl(train_loader, encoder, device, args.batch_size)

            for i in range(0, args.train_m) : 
                for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc='train M[%d/%d]' % (i, args.train_m)) :
                    uniform_input_cuda = each_batch.to(device)
                    encoded_feature_cuda = label_feature.to(device)
                    loss_m = update_mapping_ulearning(m_optimizer, uniform_input_cuda, encoded_feature_cuda, mapper)
                if args.run_test : break
                    
        if model_name in ['ulearning_point'] :     
            encoded_feature_tensor = make_encoded_feature_tensor(encoder, train_loader, device)
            linspace_tensor = make_linspace_tensor(encoded_feature_tensor)
            
            each_dim_final_loss = update_mapping_ulearning_point( mapper, linspace_tensor, \
                                           encoded_feature_tensor, args.u_lr_min, device, print_every=-1)
            loss_m = torch.max(each_dim_final_loss)

        if model_name in ['vanilla', 'pointMapping_but_aae'] :
            for i in range(0, args.train_m) : 
                for each_batch, label in tqdm.tqdm(train_loader, desc='train M[%d/%d]' % (i, args.train_m)) :
                    real_image = each_batch.to(device)
                    loss_d = update_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim)
                    loss_m = update_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim, args.std_maximize, args.std_alpha)
                    if args.run_test : break
                if args.run_test :break
        encoder.train()
           

    # train phase        
    for i in range(0, epochs):
        for each_batch, label in tqdm.tqdm(train_loader, desc='train IAAE[%d/%d]' % (i, epochs)):
            real_image = each_batch.to(device)
            loss_r = update_autoencoder(ae_optimizer, real_image, encoder, decoder)
            
            if model_name in ['vanilla', 'pointMapping_but_aae'] : 
                loss_d = update_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim)
            else : 
                loss_d = 0.
            if args.latent_layer > 0 and model_name in ['vanilla', 'pointMapping_but_aae'] : 
                loss_m =  update_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim, args.std_maximize, args.std_alpha)
            else : 
                loss_m = 0.
            if args.run_test : break
        
        if model_name == 'ulearning' and i % args.train_m_interval == 0 :
            feature_tensor_dloader = make_ulearning_dsl(train_loader, encoder, device, args.batch_size)
            for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc='train M[%d/%d]' % (i, epochs)) :
                uniform_input_cuda = each_batch.to(device)
                encoded_feature_cuda = label_feature.to(device)
                loss_m = update_mapping_ulearning(m_optimizer, uniform_input_cuda, encoded_feature_cuda, mapper)
            if args.run_test : break
                
        if model_name in ['ulearning_point'] : 
            encoded_feature_tensor = make_encoded_feature_tensor(encoder, train_loader, device)
            linspace_tensor = make_linspace_tensor(encoded_feature_tensor)
            
            each_dim_final_loss = update_mapping_ulearning_point( mapper, linspace_tensor, \
                                           encoded_feature_tensor, args.u_lr_min, device, print_every=-1)
            loss_m = torch.max(each_dim_final_loss)
            
       
        if i % save_image_interval == 0:
            for each_batch, label in tqdm.tqdm(train_loader, desc='generate image[%d/%d]' % (i, epochs)):    
                if model_name in ['vanilla', 'pointMapping_but_aae'] : 
                    sampled_images = inference_image(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device)
                if model_name == 'ulearning' :
                    sampled_images = inference_image_ulver(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device)
                if model_name == 'ulearning_point' :
                    sampled_images = inference_image_ulpver(mapper, decoder, batch_size=each_batch.size(0), latent_dim=latent_dim, device=device)
                inception_model_score.put_fake(sampled_images)    
        
        if i % save_image_interval == 0:
            
            fixed_fake_image = get_fixed_z_image_np(decoder(mapper(fixed_z)))
            
            #offload all GAN model to cpu and onload inception model to gpu
            encoder = encoder.to('cpu')
            decoder = decoder.to('cpu')
            if args.latent_layer > 0 : mapper = mapper.to('cpu')
            if model_name in ['vanilla', 'pointMapping_but_aae']: 
                discriminator = discriminator.to('cpu')
            inception_model_score.model_to(device)
            
            #generate fake images info
            inception_model_score.lazy_forward(batch_size=64, device=device, fake_forward=True)
            inception_model_score.calculate_fake_image_statistics()
            metrics, plot = inception_model_score.calculate_generative_score(feature_pca_plot=True)
            metrics.update({'IsNet feature':wandb.Image(plot)})
            
            #onload all GAN model to gpu and offload inception model to cpu
            inception_model_score.model_to('cpu')
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            if args.latent_layer > 0 : mapper = mapper.to(device)

            real_encoded_data = get_encoded_data(train_loader, encoder, device=device, size=2048)                
            mapper_test_data, mapper_out_data = make_mapper_out(model_name, mapper, latent_dim, args.latent_layer, device) 
                            
            mapper_input_out_plot =  wandb.Image(pca_kde(mapper_test_data, mapper_out_data, real_encoded_data))
            feature_kde = feature_plt_list(mapper_test_data, mapper_out_data, real_encoded_data)
            
            if model_name in ['vanilla', 'pointMapping_but_aae'] : 
                discriminator = discriminator.to(device)
            
            metrics.update({
                       "fake_image" :[wandb.Image(fixed_fake_image, caption='fixed z image')],
                       "mapper_inout(pca 1dim)" : mapper_input_out_plot,
                       "feature_kde" : [wandb.Image(plt) for plt in feature_kde],
                        'loss_r' : loss_r,
                        'loss_d': loss_d,
                        'loss_m': loss_m,
                      })
            if args.wandb : 
                wandb.log(metrics, step=i)
            
            inception_model_score.clear_fake()

    if args.wandb :  wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--save_image_interval', type=int, default=5)
    parser.add_argument('--loss_calculation_interval', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--mapper_inter_nz', type=int, default=10)
    parser.add_argument('--mapper_inter_layer', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--project_name', type=str, default='AAE_dc')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['LSUN_dining_room', 'LSUN_classroom', 'LSUN_conference', 'LSUN_churches',
                                                                    'FFHQ', 'CelebA', 'cifar10', 'mnist', 'mnist_fashion', 'emnist'])

    parser.add_argument('--model_name', type=str, default='vanilla', choices=['vanilla', 'ulearning', 'ulearning_point', \
                                                                              'pointMapping_but_aae', 'std_learning'])
    parser.add_argument('--std_maximize', type=bool, default=False)
    parser.add_argument('--std_alpha', type=float, default=0.1)
    parser.add_argument('--train_m_interval', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run_test', type=bool, default=False)
    parser.add_argument('--latent_layer', type=int, default=0)
    parser.add_argument('--AE_iter', type=int, default=0)
    parser.add_argument('--train_m', type=int, default=0)
    parser.add_argument('--u_lr_min', type=float, default=1e-4)
    parser.add_argument('--wandb', type=bool, default=False)

    args = parser.parse_args()

    main(args)
