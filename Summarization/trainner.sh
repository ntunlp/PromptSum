python -m torch.distributed.launch --nproc_per_node 1 --master_port 29565 main.py --pretrain_t5_tagger \
    --valid_size_per_gpu 16 \
    --batch_size_per_gpu 3 \
    --cuda 3 \
    --exp_id 007 \
    --concat_mode concat_left \
    --pretrain_all_weights \
    --prompt_number 300 \
    --pretrain_with_ent_chain \
    --gradient_accumulation_steps 1 \
