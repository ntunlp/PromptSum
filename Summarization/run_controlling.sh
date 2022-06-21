# pretrain_ckpt="/data/hailin/PromptSumm/006_bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/006_bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_15k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_15k/bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_330k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_330k/bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_135k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_135k/bestckpt_prompt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"

# 10-shot
# echo "start 10-shot prompt-tune_entity"
# CUDA_VISIBLE_DEVICES=0 python main.py --few_shot 100 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --num_seeds 1 --dataset_name xsum
# echo "end 10-shot prompt-tune_entity"

# echo "start 10-shot prompt-tune_summary"
# python main.py --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --num_seeds 1 --dataset_name xsum
# echo "end 10-shot prompt-tune_summary"

# echo "CONTROLLING EXPS"
# CUDA_VISIBLE_DEVICES=0 python controllability.py --dataset_name=xsum --few_shot 100 --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt

# use a bigger testset
echo "CONTROLLING EXPS"
CUDA_VISIBLE_DEVICES=0 python controllability.py --big_testset --dataset_name=xsum --few_shot 100 --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt

# nohup bash run_controlling.sh > 1.controlling 2>&1 &
# --guidance_mode target