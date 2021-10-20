python main.py --device=cuda:0 --dataset=FFHQ --model_name=vanilla --latent_dim=32 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100 --batch_size=64 --time_limit=02:50:00

python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=10 --train_m=10 --model_name=non-prior --latent_dim=32 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100 --batch_size=64 --time_limit=02:50:00

python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=10 --train_m=1 --model_name=ulearning_point --latent_dim=32 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100 --batch_size=64 --time_limit=02:50:00

python main.py --device=cuda:0 --dataset=FFHQ --AE_iter=10 --train_m=10 --model_name=pointMapping_but_aae --latent_dim=32 --img_size=128 --epochs=100 --wandb=True --save_image_interval=100 --batch_size=64 --time_limit=02:50:00