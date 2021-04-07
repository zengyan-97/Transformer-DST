export CUDA_VISIBLE_DEVICES=0
DATASET_DIR="./data/v21/"
SAVE_DIR="./outputs/dt_e15_b16/"

python3 train.py --use_dt_only --use_one_optim --n_epochs 15 --batch_size 16 --data_root $DATASET_DIR --save_dir $SAVE_DIR --op_code '4'
