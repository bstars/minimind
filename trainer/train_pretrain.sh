
# colab pretraining

# !python train_pretrain.py \
#     --epochs 3 \
#     --batch_size 128 \
#     --learning_rate 0.0005 \
#     --accumulation_steps 2 \
#     --grad_clip 1.0 \
#     --weight_decay 0.01 \
#     --max_seq_len 350 \
#     --print_every 10 \
#     --data_path ../../data/pretrain_t2t_mini.jsonl \
#     --compile