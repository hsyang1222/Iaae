import generative_model_score
inception_model_score = generative_model_score.GenerativeModelScore()
from datetime import datetime
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
    
    fixed_z = make_fixed_z(model_name, latent_dim, device)

    image_shape = [3, img_size, img_size]
    
    time_limit_sec = timeparse(args.time_limit)
    time_start_run = time.time()    
    

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
    if model_name in ['vanilla'] :
        args.mapper_inter_layer = 0 
        
        
    if model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior']:
        encoder = Encoder(latent_dim, image_shape).to(device)
        decoder = Decoder(latent_dim, image_shape).to(device)
        discriminator = Discriminator(latent_dim).to(device)
        ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        
    elif model_name in ['ulearning', 'ulearning_point', 'mimic_at_last', 'mimic'] : 
        encoder = Encoder(latent_dim, image_shape, sigmoid=True).to(device)
        decoder = Decoder(latent_dim, image_shape).to(device)
        discriminator = None
        d_optimizer = None
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
        train_loader = get_ffhq_thumbnails(batch_size, img_size)
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
        inception_model_score.lazy_forward(batch_size=256, device=device, real_forward=True)
        inception_model_score.calculate_real_image_statistics()
        #save real images info for next experiments
        inception_model_score.save_real_images_info('../../inception_model_info/' + real_images_info_file_name)
        #offload inception_model
        inception_model_score.model_to('cpu')

   
    if args.mapper_inter_layer > 0 : 
        if model_name in ['ulearning_point', 'mimic_at_last']:
            mapper = EachLatentMapping(nz=args.latent_dim, inter_nz=args.mapper_inter_nz, linear_num=args.mapper_inter_layer).to(device)
            m_optimizer = None
        elif model_name in [ 'pointMapping_but_aae']:
            mapper = EachLatentMapping(nz=args.latent_dim, inter_nz=args.mapper_inter_nz, linear_num=args.mapper_inter_layer).to(device)
            m_optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
        elif model_name in ['ulearning', 'non-prior'] : 
            mapper = Mapping(args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(device)
            m_optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
        elif model_name in ['mimic'] : 
            mapper = Mimic(args.latent_dim, args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(device)
            m_optimizer = torch.optim.Adam(mapper.parameters(), lr=lr, weight_decay=1e-3)
    else :
        # case vanilla and there is no mapper
        mapper = lambda x : x
        m_optimizer = None

    if args.load_netE!='' : load_model(encoder, args.load_netE) 
    if args.load_netM!='' : load_model(mapper, args.load_netM)   
    if args.load_netD!='' : load_model(decoder, args.load_netD)   
        
        
    AE_pretrain(args, train_loader, device, ae_optimizer, encoder, decoder)    
    
    M_pretrain(args, train_loader, device, d_optimizer, m_optimizer, mapper, encoder, discriminator)

    # train phase
    i=0
    loss_log={}
    for i in range(1, epochs+1):
        loss_log = train_main(args, train_loader, i, device, ae_optimizer, m_optimizer, d_optimizer, encoder, decoder, mapper, discriminator)
        
        if check_time_over(time_start_run, time_limit_sec) == True :
            break
            
        if i % save_image_interval == 0:
            insert_sample_image_inception(args, i, epochs, train_loader, mapper, decoder, inception_model_score)
            matric = gen_matric(wandb, args, train_loader, encoder, mapper, decoder, discriminator, inception_model_score)
            loss_log.update(matric)
        if args.wandb : 
            wandb_update(wandb, i, args, train_loader, encoder, mapper, decoder, device, fixed_z, loss_log)
        else : 
            print(loss_log)
            
        if i % args.save_model_every == 0 :
            now_time = str(datetime.now())
            save_model([encoder, mapper, decoder], [now_time+'.netE', now_time+'.netM', now_time+'.netD'])
            
    #make last matric        
    if model_name in ['mimic_at_last'] :
        M_train_at_last(args, train_loader, device, d_optimizer, m_optimizer, mapper, encoder, discriminator)
    print("time limit over")
    insert_sample_image_inception(args, i, epochs, train_loader, mapper, decoder, inception_model_score)
    matric = gen_matric(wandb, args, train_loader, encoder, mapper, decoder, discriminator, inception_model_score)
    loss_log.update(matric)
    wandb_update(wandb, i, args, train_loader, encoder, mapper, decoder, device, fixed_z, loss_log)

    now_time = str(datetime.now())
    save_model([encoder, mapper, decoder], [now_time+'.netE', now_time+'.netM', now_time+'.netD'])
    

    if args.wandb :  wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--save_image_interval', type=int, default=10)
    parser.add_argument('--loss_calculation_interval', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--mapper_inter_nz', type=int, default=10)
    parser.add_argument('--mapper_inter_layer', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--project_name', type=str, default='AAE_compare')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['LSUN_dining_room', 'LSUN_classroom', 'LSUN_conference', 'LSUN_churches',
                                                                    'FFHQ', 'CelebA', 'cifar10', 'mnist', 'mnist_fashion', 'emnist'])

    parser.add_argument('--model_name', type=str, default='vanilla', choices=['vanilla', 'ulearning', 'ulearning_point', \
                                                                              'pointMapping_but_aae', 'non-prior', \
                                                                              'mimic_at_last', 'mimic'])
    parser.add_argument('--std_maximize', type=bool, default=False)
    parser.add_argument('--std_alpha', type=float, default=0.1)
    parser.add_argument('--train_m_interval', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run_test', type=bool, default=False)
    parser.add_argument('--AE_iter', type=int, default=0)
    parser.add_argument('--pretrain_m', type=int, default=0)
    parser.add_argument('--train_m', type=int, default=1)
    parser.add_argument('--u_lr_min', type=float, default=1e-4)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--save_model_every', type=int, default=25)
    parser.add_argument('--load_netE', type=str, default='')
    parser.add_argument('--load_netM', type=str, default='')
    parser.add_argument('--load_netD', type=str, default='')
    
    parser.add_argument('--m_wdecay', type=float, default=1e-3)
    parser.add_argument('--m_gmma', type=float, default=0.9995)
    parser.add_argument('--m_blr', type=float, default=0.1e-2)
    parser.add_argument('--m_batch_size', type=int, default=2048)
    
    parser.add_argument('--M_feature', type=str, default='')
    
    import socket
    parser.add_argument('--host_name', type=str, default=str(socket.gethostname()))
    parser.add_argument('--time_limit', type=str, default='', help="hour:min:sec")

    args = parser.parse_args()

    main(args)
