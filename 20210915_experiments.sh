python main.py --device=cuda:0 --dataset=FFHQ --mapper_inter_layer=2 --AE_iter=10 --train_m=10 --model_name=ulearning --latent_dim=64 --img_size=128 --wandb=True 

python main.py --device=cuda:0 --dataset=FFHQ --mapper_inter_layer=10 --AE_iter=10 --train_m=10 --model_name=ulearning --latent_dim=64 --img_size=128 --wandb=True 

python main.py --device=cuda:0 --dataset=FFHQ --mapper_inter_layer=10 --AE_iter=10 --train_m=10 --model_name=pointMapping_but_aae --latent_dim=64 --img_size=128 --wandb=True 

python main.py --device=cuda:0 --dataset=FFHQ --mapper_inter_layer=10 --AE_iter=10 --train_m=10 --model_name=ulearning_point --latent_dim=64 --img_size=128 --wandb=True 