learnrate=(5e-1)
for onerate in ${learnrate[@]}
do
  echo "------------------------------"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 29510 main.py \
          --cuda 0,1 \
          --lr $onerate \
          --optimizer Adafactor \
          --weight_decay 1e-5 \
          --max_grad_norm 1.0 \
          --batch_size_per_gpu 4 \
          --valid_size_per_gpu 16 \
          --test_size_per_gpu 16 \
          --gradient_accumulation_steps 4 \
          --max_epoch 5 \
          --num_workers 4 \
          --log_step 10 \
          --concat_mode 'right_concat'  \
          --save_dir t5summ_right_ckpt_v011  \
          --guidance_mode oracle \
          --seed 42 \
          --model T5MixPrompt \
          --model_name google/t5-v1_1-base \
          --adam_epsilon 1e-8 \
          --warmup_steps 0.01 \
          --use_lm_adapted 1 \
          --lm_adapted_path /export/home/prompting/lm_adapted_models/t5.1.1.lm100k.base/pytorch_model.bin\
          --prompt_length 100 \
          --prompt_length_task 100\
          --ifckpt_onlymodel 1\
          # --dataset_name cnn_dailymail \
          # --dataset_version 3.0.0 \
          # --summary_key highlights \
          # --dataset_cache_dir /export/home/cache/ \
          # --load_ckpt 0 \
          # --ckpt_path t5ner_ckpt/t5nerlarge_full_right_ckpt_v038/ckptofT5ner_21114\
done


