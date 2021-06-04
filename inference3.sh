python main.py --dataset=mnist --model_name=mod2_var --device=cuda:3 --batch_size=1024
python main.py --dataset=mnist --model_name=latent_mapping --device=cuda:3 --batch_size=1024
python main.py --dataset=mnist --model_name=gme_inference --device=cuda:3 --batch_size=1024

python main.py --dataset=FFHQ --model_name=mod2_var --device=cuda:3 --batch_size=1024
python main.py --dataset=FFHQ --model_name=gme_inference --device=cuda:3 --batch_size=1024

