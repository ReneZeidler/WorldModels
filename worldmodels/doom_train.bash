CUDA_VISIBLE_DEVICES=0 python3 train_vae.py -c configs/doom.config
CUDA_VISIBLE_DEVICES=0 python3 encode_samples.py -c configs/doom.config
CUDA_VISIBLE_DEVICES=0 python3 train_rnn.py -c configs/doom.config
CUDA_VISIBLE_DEVICES=-1 python3 train_controller.py -c configs/doom.config
