pretrain_ckpt="/data/hailin/PromptSumm/006_bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/006_bestckpt_prompt"


# 10-shot
echo "start 10-shot prompt-tune_entity"
python main.py --few_shot 10 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --num_seeds 1 --dataset_name xsum
echo "end 10-shot prompt-tune_entity"

echo "start 10-shot prompt-tune_summary"
#python main.py --few_shot 10 --finetune_summary --infer_val_entities --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
python main.py --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --save_model True --save_model_path /data/ruochen/DATASETS/PromptSumm/xsum/10/seed_0/best_ckpt --num_seeds 1 --dataset_name xsum
echo "end 10-shot prompt-tune_summary"

echo "CONTROLLING EXPS"
CUDA_VISIBLE_DEVICES=4 python controllability.py --dataset_name=xsum

# nohup bash run_controlling.sh > 1.controlling 2>&1 &