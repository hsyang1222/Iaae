python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=30 --train_m=1 --model_name=ulearning_point --latent_dim=128 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100

python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=30 --train_m=1 --model_name=ulearning_point --latent_dim=256 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100

python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=30 --train_m=1 --model_name=ulearning_point --latent_dim=512 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100

python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=30 --train_m=1 --model_name=ulearning_point --latent_dim=1024 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100