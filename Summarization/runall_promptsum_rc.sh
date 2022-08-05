#pretrain_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_full_model_114k"
#pretrain_prompt_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_prompt_114k"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_15k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_15k/bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/013_ent_135k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/013_ent_135k/bestckpt_prompt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"


# ## 10-shot
# echo "start 10-shot prompt-tune_entity"
# # python main.py --few_shot 10 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --dataset_name ccdv/cnn_dailymail
# echo "end 10-shot prompt-tune_entity"

# echo "start 10-shot prompt-tune_summary"
# # python main.py --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --full_testset --dataset_name ccdv/cnn_dailymail
# python main.py --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --dataset_name ccdv/cnn_dailymail
# echo "end 10-shot prompt-tune_summary"

# EVAL
# echo "start 10-shot prompt-tune_summary EVALUATION"
# CUDA_VISIBLE_DEVICES=2 python main.py --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --full_testset --dataset_name ccdv/cnn_dailymail

# ## 64-shot
# echo "start 64-shot prompt-tune_entity"
# python main.py --few_shot 64 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --dataset_name ccdv/cnn_dailymail
# echo "end 64-shot prompt-tune_entity"

# echo "start 64-shot prompt-tune_summary"
# python main.py --few_shot 64 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --dataset_name ccdv/cnn_dailymail
# echo "end 64-shot prompt-tune_summary"

# # EVAL
# python main.py --few_shot 64 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --full_testset --dataset_name ccdv/cnn_dailymail

# ## 100-shot
# echo "start 100-shot prompt-tune_entity"
# python main.py --few_shot 100 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --dataset_name ccdv/cnn_dailymail
# echo "end 100-shot prompt-tune_entity"

# echo "start 100-shot prompt-tune_summary"
# python main.py --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --dataset_name ccdv/cnn_dailymail
# echo "end 100-shot prompt-tune_summary"

# # EVAL
# echo "start 100-shot prompt-tune_summary EVALUATION"
# CUDA_VISIBLE_DEVICES=4 python main.py --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --full_testset --dataset_name ccdv/cnn_dailymail

# echo "start 10-shot prompt-tune_summary CNNDM evaluation ORACLE"
# CUDA_VISIBLE_DEVICES=1 python main.py --testing --guidance_mode target --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0  --dataset_name ccdv/cnn_dailymail
# # nohup bash runall_promptsum_rc.sh > oracle_10_cnndm_log 2>&1 &

echo "start 100-shot prompt-tune_summary CNNDM evaluation ORACLE"
CUDA_VISIBLE_DEVICES=0 python main.py --testing --guidance_mode target --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --dataset_name ccdv/cnn_dailymail
# # # nohup bash runall_promptsum_rc.sh > oracle_100_cnndm_log 2>&1 &

# echo "start 10-shot prompt-tune_summary XSUM evaluation ORACLE"
# CUDA_VISIBLE_DEVICES=2 python main.py --testing --guidance_mode target --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --dataset_name xsum
# # nohup bash runall_promptsum_rc.sh > oracle_10_xsum_log 2>&1 &

# echo "start 100-shot prompt-tune_summary XSUM evaluation ORACLE"
# CUDA_VISIBLE_DEVICES=5 python main.py --testing --guidance_mode target --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --dataset_name xsum
#  --full_testset --use_t5_tagger --guidance_mode target --testing --finetune_entity
# nohup bash runall_promptsum_rc.sh > oracle_100_xsum_log 2>&1 &