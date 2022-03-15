learnrate=(5e-1)
for onerate in ${learnrate[@]}
do
  echo "------------------------------"
  python -m torch.distributed.launch --nproc_per_node 1 --master_port 29514 main.py \
          --cuda 0 \
          --lr $onerate \
          --optimizer Adafactor \
          --weight_decay 1e-5 \
          --max_grad_norm 1.0 \
          --batch_size_per_gpu 4 \
          --valid_size_per_gpu 32 \
          --test_size_per_gpu 32 \
          --gradient_accumulation_steps 4 \
          --max_epoch 5 \
          --num_workers 0 \
          --log_step 10 \
          --eval_step 10\
          --eval_start_epoch 0\
          --eval_epoch 1\
          --concat_mode 'right_concat'  \
          --save_dir t5summ_right_ckpt_v017  \
          --guidance_mode normal \
          --seed 42 \
          --model T5MixPrompt \
          --model_name google/t5-v1_1-base \
          --adam_epsilon 1e-8 \
          --warmup_steps 0.01 \
          --use_lm_adapted 1 \
          --lm_adapted_path /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin\
          --prompt_length 100 \
          --prompt_length_task 100\
          --ifckpt_onlymodel 1\
          --guidance_type ents \
          --max_target_length 128 \
          --max_guidance_len 100 \
          # --dataset_name cnn_dailymail \
          # --dataset_version 3.0.0 \
          # --summary_key highlights \
          # --text_key article \
          # --dataset_cache_dir /export/home/cache/ \
          # --load_ckpt 0 \
          # --ckpt_path t5ner_ckpt/t5nerlarge_full_right_ckpt_v038/ckptofT5ner_21114\
done


