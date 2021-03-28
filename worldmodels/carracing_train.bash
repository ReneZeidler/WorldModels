CUDA_VISIBLE_DEVICES=0 python3 train_vae.py -c configs/carracing.config
CUDA_VISIBLE_DEVICES=0 python3 encode_samples.py -c configs/carracing.config
CUDA_VISIBLE_DEVICES=0 python3 train_rnn.py -c configs/carracing.config
CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 train_controller.py -c configs/carracing.config
