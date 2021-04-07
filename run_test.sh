export CUDA_VISIBLE_DEVICES=0

DATASET_DIR="./data/v21/"
SAVE_DIR="./outputs/dt_e15_b16/"

# --load_epoch 10-11-12-13-14-15

python3 test.py --use_dt_only --use_one_optim --random_seed 42 --load_epoch 15 --batch_size 1 --beam_size 1 --data_root $DATASET_DIR --save_dir $SAVE_DIR --op_code '4'
