learnrate=(1e-3)
for onerate in ${learnrate[@]}
do
      echo "------------------------------"
      python -m torch.distributed.launch --nproc_per_node 2 --master_port 29512 main.py \
              --cuda 0,1 \
              --lr $onerate \
              --batch_size_per_gpu 8 \
              --valid_size_per_gpu 32 \
              --test_size_per_gpu 16 \
              --gradient_accumulation_steps 1 \
              --max_epoch 200 \
              --num_workers 4 \
              --log_step 20 \
              --save_step 18000 \
              --eval_step 100 \
              --save_dir t5ner_fewshot_finetune_ckpt_v054 \
              --seed 42 \
              --model T5Finetune \
              --model_name google/t5-v1_1-base \
              --max_length 128 \
              --adam_epsilon 1e-8 \
              --warmup_steps 0.01 \
              --load_ckpt 0 \
              --use_lm_adapted 1 \
              --lm_adapted_path /export/home/prompting/lm_adapted_models/t5.1.1.lm100k.base/pytorch_model.bin\
              --ifckpt_onlymodel 1\

    echo "++++++++++++++++++++++++++++++"
    #ps aux | grep main_prompt_adafactor.py | awk '{print $2}' | xargs kill -9
    #--ckpt_path t5ner_ckpt/t5nerlarge_full_mixed_right_ckpt_v039/ckptofT5ner_56240\
done


