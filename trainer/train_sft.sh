# smoke test
python train_sft.py \
    --batch_size 2 \
    --max_seq_len 512 \
    --learning_rate 0.0003 \
    --epochs 60 \
    --accumulation_steps 2 \
    --subset 7 \
    --print_every 1 \
    --save_weight smoke_test.pth \
    --weight_decay 0.000