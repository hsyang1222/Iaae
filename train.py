import torch
from torch.autograd import Variable
import numpy as np
import tqdm
from utils import *


def AE_pretrain(args, train_loader, device, ae_optimizer, encoder, decoder):
    loss_r_sum = -1.
    
    for i in range(0, args.AE_iter) :
        last_loss = loss_r_sum
        loss_r_sum = 0.
        for each_batch, label in tqdm.tqdm(train_loader, desc='pretrain AE[%d/%d](l=%.04f)' % (i, args.AE_iter, last_loss)) :
            real_image = each_batch.to(device)
            loss_r = update_autoencoder(ae_optimizer, real_image, encoder, decoder)
            if args.run_test : break
            loss_r_sum += loss_r
        #print(i, loss_r.item())
        if args.run_test : break
        
def M_pretrain(args, train_loader, device, d_optimizer, m_optimizer, mapper, encoder, discriminator) :
    loss_m_sum=-1.
    latent_dim = args.latent_dim
    #pretrain M layer
    model_name = args.model_name
    if args.mapper_inter_layer > 0 :
        encoder.eval()
            
        if model_name in ['ulearning'] : 

            feature_tensor_dloader = make_ulearning_dsl(train_loader, encoder, device, args.batch_size)

            for i in range(0, args.pretrain_m) : 
                for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc='pretrain M [%d/%d]' % (i, args.pretrain_m)) :
                    uniform_input_cuda = each_batch.to(device)
                    encoded_feature_cuda = label_feature.to(device)
                    loss_m = update_mapping_ulearning(m_optimizer, uniform_input_cuda, encoded_feature_cuda, mapper)
                if args.run_test : break
                    
        if model_name in ['ulearning_point'] :     
            encoded_feature_tensor = make_encoded_feature_tensor(encoder, train_loader, device)
            linspace_tensor = make_linspace_tensor(encoded_feature_tensor)
            
            each_dim_final_loss = update_mapping_ulearning_point( mapper, linspace_tensor, \
                                           encoded_feature_tensor, args.u_lr_min, device, print_every=1e+4)
            loss_m = torch.max(each_dim_final_loss)

        if model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior'] :
            for i in range(0, args.pretrain_m) : 
                for each_batch, label in tqdm.tqdm(train_loader, desc='pretrain M [%d/%d]' % (i, args.pretrain_m)) :
                    real_image = each_batch.to(device)
                    loss_d = update_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim)
                    loss_m = update_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim, args.std_maximize, args.std_alpha)
                    if args.run_test : break
                if args.run_test :break
                
        if model_name in ['mimic'] :
            feature_tensor_dloader = make_ulearning_dsl(train_loader, encoder, device, args.batch_size)
            
            for i in range(0, args.pretrain_m) : 
                last_loss = loss_m_sum
                loss_m_sum = 0.
                for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc='pretrain M [%d/%d](l=%.04f)' % (i, args.pretrain_m, last_loss)) :
                    uniform_input_cuda = each_batch.to(device)
                    sortedencoded_feature_cuda = label_feature.to(device)
                    loss_m = update_mimic(m_optimizer, uniform_input_cuda, sortedencoded_feature_cuda, mapper)
                    loss_m_sum+=loss_m
                if args.run_test : break   
            
            encoder.train()
            return feature_tensor_dloader, loss_m_sum
        
        if model_name in ['mimic+non-prior'] :
            feature_tensor_dloader = make_ulearning_dsl(train_loader, encoder, device, args.batch_size)
            
            for i in range(0, args.pretrain_m) : 
                last_loss = loss_m_sum
                loss_m_sum = 0.
                
                #train mapper by mimic style
                for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc='pretrain M-mimic[%d/%d](l=%.04f)' % (i, args.pretrain_m, last_loss)) :
                    
                    #shuffle in each dim
                    perm_index = torch.stack([torch.randperm(each_batch.size(0)) for i in range(each_batch.size(1))],dim=1)

                    uniform_input_cuda = each_batch[perm_index].to(device)
                    sortedencoded_feature_cuda = label_feature[perm_index].to(device)
                    
                    loss_m = update_mimic(m_optimizer, uniform_input_cuda, sortedencoded_feature_cuda, mapper)
                    loss_m_sum+=loss_m
                    if args.run_test : break   
                
                #train mapper by discriminator style
                for each_batch, label in tqdm.tqdm(train_loader, desc='pretrain M-discr[%d/%d]' % (i, args.pretrain_m)) :
                    real_image = each_batch.to(device)
                    loss_d = update_linspace_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim)
                    loss_m = update_linspace_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim)
                    if args.run_test : break
                    
                if args.run_test :break
            
            encoder.train()
            return feature_tensor_dloader, loss_m_sum        
            
        encoder.train()
        return loss_m_sum
        

def update_mimic(m_optimizer, x_data_cuda, target_data_cuda, net) :
    mse = torch.nn.MSELoss(reduction='sum')
    
    predict = net(x_data_cuda)
    loss=mse(predict, target_data_cuda)

    m_optimizer.zero_grad()
    loss.backward()
    m_optimizer.step()

    return loss.item()

        
        
def M_train_at_last(args, train_loader, device, d_optimizer, m_optimizer, mapper, encoder, discriminator) : 
    latent_dim = args.latent_dim
    #pretrain M layer
    model_name = args.model_name
    
    if args.M_feature == '':
        encoded_feature_tensor = make_encoded_feature_tensor(encoder, train_loader, device)
    else:
        print('use saved encoded_feature_tensor')
        encoded_feature_tensor = torch.load(args.M_feature)
    
    linspace_tensor = make_linspace_tensor(encoded_feature_tensor)

    each_dim_final_loss = update_mapping_ulearning_point( mapper, linspace_tensor, \
                               encoded_feature_tensor, args.u_lr_min, device, print_every=1e+3, \
                                                        wdecay=args.m_wdecay, gamma=args.m_gmma, base_lr=args.m_blr, batch_size=args.m_batch_size)
    loss_m = torch.max(each_dim_final_loss)
        
        
def train_main(args, train_loader, i, device, ae_optimizer, m_optimizer, d_optimizer, encoder, decoder, mapper, discriminator):
    latent_dim = args.latent_dim
    epochs = args.epochs
    model_name = args.model_name
    
    loss_d = 0.
    loss_m = 0.
    loss_r = 0.
    loss_md = 0.
    
    encoded_feature_list = []
    for each_batch, label in tqdm.tqdm(train_loader, desc='train AE[%d/%d]' % (i, epochs)):
        real_image = each_batch.to(device)
        
        if model_name in ['mimic', 'mimic+non-prior'] : 
            loss_r, encoded_feature = update_autoencoder(ae_optimizer, real_image, encoder, decoder, return_encoded_feature=True)
            encoded_feature_list.append(encoded_feature)
        else:
            loss_r = update_autoencoder(ae_optimizer, real_image, encoder, decoder)

        if model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior'] : 
            loss_d = update_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim)
        else : 
            loss_d = 0.
        if args.mapper_inter_layer > 0 and model_name in ['vanilla', 'pointMapping_but_aae', 'non-prior'] : 
            loss_m =  update_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim, args.std_maximize, args.std_alpha)
        else : 
            loss_m = 0.
        if args.run_test : break
        
    if model_name == 'ulearning' and i % args.train_m_interval == 0 :
        feature_tensor_dloader = make_ulearning_dsl(train_loader, encoder, device, args.batch_size)
        for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc='train M    [%d/%d]' % (i, epochs)) :
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
        
    if model_name in ['mimic'] :
        feature_tensor_dloader = encoded_feature_to_dl(torch.cat(encoded_feature_list), args.batch_size)
        loss_m_sum = -1.
        for m_i in range(0, args.train_m) : 
            last_loss = loss_m_sum
            loss_m_sum = 0.
            if args.train_m > 1 : 
                desc='train M [%d/%d/%d](l=%.04f)' % (m_i, args.train_m, i, last_loss)
            else:
                desc='train M [%d/%d]' % (i,epochs)
            for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc=desc) :
                uniform_input_cuda = each_batch.to(device)
                sortedencoded_feature_cuda = label_feature.to(device)
                loss_m = update_mimic(m_optimizer, uniform_input_cuda, sortedencoded_feature_cuda, mapper)
                loss_m_sum+=loss_m
                if args.run_test : break   
            if args.run_test : break   
        loss_m = loss_m_sum
    elif model_name in ['mimic+non-prior'] : 
        feature_tensor_dloader = encoded_feature_to_dl(torch.cat(encoded_feature_list), args.batch_size)
        loss_m_sum = -1.
        #train m by mimic
        for m_i in range(0, args.train_m) : 
            last_loss = loss_m_sum
            loss_m_sum = 0.
            if args.train_m > 1 : 
                desc='train M-mimic[%d/%d, %d/%d](l=%.04f)' % (m_i, args.train_m, i, epochs, last_loss)
            else:
                desc='train M-mimic[%d/%d]' % (i,epochs)
            for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader, desc=desc) :
                
                #shuffle in each dim
                perm_index = torch.stack([torch.randperm(each_batch.size(0)) for i in range(each_batch.size(1))],dim=1)
                uniform_input_cuda = each_batch[perm_index].to(device)
                sortedencoded_feature_cuda = label_feature[perm_index].to(device)
                loss_m = update_mimic(m_optimizer, uniform_input_cuda, sortedencoded_feature_cuda, mapper)
                loss_m_sum+=loss_m
                if args.run_test : break   
            if args.run_test : break   
        
        loss_md_sum = 0.
        loss_d_sum= 0.
        #train m by discriminator
        for each_batch, label in tqdm.tqdm(train_loader, desc='train M-disrm[%d/%d]' % (i, epochs)) :
            real_image = each_batch.to(device)
            loss_d = update_linspace_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim)
            loss_m = update_linspace_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim)
            loss_d_sum += loss_d
            loss_md_sum += loss_m
            if args.run_test : break
        
        loss_m = loss_m_sum        
        loss_d = loss_d_sum
        loss_md = loss_md_sum

        
    loss_log = {
        'loss_d' : loss_d,
        'loss_m' : loss_m,
        'loss_r' : loss_r,
        'loss_md' : loss_md,
      #  'feature_tensor_dloader' : feature_tensor_dloader,
    }
    
    return loss_log
    
    

def update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder, return_encoded_feature=False):
    ae_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    X_decoded = decoder(z_posterior)
    pixelwise_loss = torch.nn.L1Loss(reduction='sum')
    r_loss = pixelwise_loss(X_decoded, X_train_batch)
    r_loss.backward()
    ae_optimizer.step()
    if return_encoded_feature:
        return r_loss.item(), z_posterior.detach().cpu()
    return r_loss.item()

def update_mapping_ulearning(m_optimizer, uniform_input_cuda, encoded_feature_cuda, mapper) :
    mse = torch.nn.MSELoss()
    predict_feature = mapper(uniform_input_cuda)
    loss_m = mse(predict_feature, encoded_feature_cuda)
    m_optimizer.zero_grad()
    loss_m.backward()
    m_optimizer.step()
    
    return loss_m.item()

def update_mapping_ulearning_point(mapper, x, target, go_under_loss, device, print_every=-1, \
                                   train_max=1e+5, wdecay=1e-3, gamma=0.9995, base_lr=1e-2, batch_size=2048) :
    mse = torch.nn.MSELoss(reduction='sum')
    nz = mapper.nz
    loss_list = torch.zeros(nz)
    
    loss_list = torch.zeros(nz)
    
    from torch.utils.data import  TensorDataset, DataLoader
    import time
    
    for each_nz in tqdm.tqdm(range(nz), desc='train M_p'):

        mapper.zero_grad()
        
        focus_x = x[:,each_nz].view(-1,1).to(device)
        focus_mapper = mapper.pm_list[each_nz]
        focus_optimizer = torch.optim.Adam(focus_mapper.parameters(), lr=base_lr, weight_decay=wdecay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(focus_optimizer, gamma=gamma)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(focus_optimizer, T_max=1e+2, eta_min=0)
        focus_target = target[:,each_nz].sort()[0].view(-1,1).to(device)

        dataset = TensorDataset(focus_x, focus_target)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        last_loss = go_under_loss + 1
        step = 0

        start_time = time.time()
        time_limit = 110 #second
        
        while last_loss >= go_under_loss:
            last_loss = 0.
            for load_x, load_target in dataloader : 
                predict = focus_mapper(load_x)
                loss = mse(predict, load_target)

                focus_optimizer.zero_grad()
                loss.backward()
                focus_optimizer.step()
                
                last_loss+=loss.item()
            last_loss = last_loss/len(dataset)
            scheduler.step()

            step += 1
            
            if time.time() - start_time > time_limit : 
                print("over time limit 110 sec")
                print("dim:%d, step:%d, go to under loss : %f, current loss:%f, lr:%f" % \
                      (each_nz, step, go_under_loss, last_loss, scheduler.get_last_lr()[0]))
                break
                
            if print_every>0 and (step % print_every == 0 or last_loss < go_under_loss):
                print("dim:%d, step:%d, go to under loss : %f, current loss:%f, lr:%f" % \
                      (each_nz, step, go_under_loss, last_loss, scheduler.get_last_lr()[0]))

                '''
                predict_encoded = predict.detach().cpu()

                plt.plot(focus_x.cpu(), focus_target.cpu(), label='E(x)')
                plt.plot(focus_x.cpu(), predict_encoded, label='M(0~1)')
                plt.xlabel("input feature ~ U[1,0]")
                plt.ylabel('latent z')
                plt.title("dim:%d, step:%d, loss:%f, lr=%f" % (each_nz, step, last_loss.item(), scheduler.get_last_lr()[0]))
                plt.legend()
                plt.show()

                sns.kdeplot(focus_target.detach().cpu().flatten(), label='E(x)')
                sns.kdeplot(predict_encoded.flatten(), label='M(0~1)')
                plt.title("dim:%d, step:%d, loss:%f, lr=%f" % (each_nz, step, last_loss.item(), scheduler.get_last_lr()[0]))
                plt.legend()
                plt.show()
                '''
            if step > train_max :
                break
        
    return loss_list

def update_linspace_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim) :
    
    batch_size = real_image.size(0)
    device = real_image.device
    bce = torch.nn.BCELoss(reduction='sum')
    
    label_one = torch.ones(batch_size, 1, device=device)
    label_zero = torch.zeros(batch_size, 1, device=device)
    
    encoded_real = encoder(real_image)
    d_predict_encoded_real = discriminator(encoded_real)    
    loss_d_e_real = bce(d_predict_encoded_real, label_one)
    
    linspace_noise = torch.rand(batch_size, latent_dim, device=device) * 2 -1 
    nonprior_map_out = mapper(linspace_noise)
    
    d_predict_mapping_fake = discriminator(nonprior_map_out)
    loss_d_m_fake = bce(d_predict_mapping_fake, label_zero)
    
    d_optimizer.zero_grad()    
    loss_d = loss_d_e_real + loss_d_m_fake
    loss_d.backward()
    d_optimizer.step()
    
    return loss_d.item()

def update_discriminator(d_optimizer, real_image, encoder, mapper, discriminator, latent_dim) :
    
    batch_size = real_image.size(0)
    device = real_image.device
    bce = torch.nn.BCELoss()
    
    label_one = torch.ones(batch_size, 1, device=device)
    label_zero = torch.zeros(batch_size, 1, device=device)
    
    encoded_real = encoder(real_image)
    d_predict_encoded_real = discriminator(encoded_real)    
    loss_d_e_real = bce(d_predict_encoded_real, label_one)
    
    gaussian_noise = torch.randn(batch_size, latent_dim, device=device)
    mapping_fake = mapper(gaussian_noise)
    d_predict_mapping_fake = discriminator(mapping_fake)
    loss_d_m_fake = bce(d_predict_mapping_fake, label_zero)
    
    d_optimizer.zero_grad()    
    loss_d = loss_d_e_real + loss_d_m_fake
    loss_d.backward()
    d_optimizer.step()
    
    return loss_d.item()


def update_linspace_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim) : 
    batch_size = real_image.size(0)
    device = real_image.device
    bce = torch.nn.BCELoss(reduction='sum')
    
    label_one = torch.ones(batch_size, 1, device=device)
    
    linspace_noise = torch.rand(batch_size, latent_dim, device=device) * 2 -1
    nonprior_map_out = mapper(linspace_noise)
    d_predict_mapping_fake = discriminator(nonprior_map_out)
    loss_d_m_fake = bce(d_predict_mapping_fake, label_one)
    
    m_optimizer.zero_grad()
    loss_m = loss_d_m_fake
    loss_m.backward()
    m_optimizer.step()
    
    return loss_m.item()    
    

def update_mapping(m_optimizer, real_image, mapper, discriminator, latent_dim, std_maximize=False, std_alpha=0.1) : 
    batch_size = real_image.size(0)
    device = real_image.device
    bce = torch.nn.BCELoss()
    
    label_one = torch.ones(batch_size, 1, device=device)
    
    gaussian_noise = torch.randn(batch_size, latent_dim, device=device)
    mapping_fake = mapper(gaussian_noise)
    d_predict_mapping_fake = discriminator(mapping_fake)
    loss_d_m_fake = bce(d_predict_mapping_fake, label_one)
    
    
    m_optimizer.zero_grad()
    loss_m = loss_d_m_fake
    loss_m.backward()
    m_optimizer.step()
    
    return loss_m.item()
    
    
    


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
    
    real_label = torch.ones((batch_size,1), device=device)
    predict_about_real = discriminator(X_train_batch)
    loss_about_real = bce(predict_about_real, real_label)
    
    fake_label = torch.zeros((batch_size,1), device=device)
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
    
    real_label = torch.ones((batch_size,1), device=device)
    
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