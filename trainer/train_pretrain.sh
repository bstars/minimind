python train_pretrain.py \
    --batch_size 4 \
    --max_seq_len 100 \
    --epochs 40 \
    --accumulation_steps 2 \
    --subset 17 \
    --print_every 1

# python train_pretrain.py \
#     --epochs 2 \
#     --batch_size 32 \
#     --learning_rate 0.0005 \
#     --accumulation_steps 8 \
#     --grad_clip 1.0 \
#     --weight_decay 0.01 \
#     --max_seq_len 350 \
#     --print_every 10