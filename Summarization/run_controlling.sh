### dataset
dataset="xsum"
k_shot="10"


### backbone model
### T5-large backbone
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


# k-shot
echo "start k-shot prompt-tune_entity"
python main.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
echo "end k-shot prompt-tune_entity"

echo "start k-shot prompt-tune_summary"
python main.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
echo "end k-shot prompt-tune_summary"

echo "start CONTROLLING experiments"
# train & val
python controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# test
#python controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --big_testset
echo "end CONTROLLING experiments"

# use a bigger testset
# echo "CONTROLLING EXPS FOR CNNDM - NO COUNTERFACTUAL TRAINING"
# CUDA_VISIBLE_DEVICES=7 python controllability.py --big_testset --dataset_name=xsum --few_shot 100 --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --dataset_name ccdv/cnn_dailymail
#  --big_testset
# nohup bash run_controlling.sh > 1.controlling 2>&1 &
# --guidance_mode target


echo "start 100-shot prompt-tune_summary for cnndm with counterfactual training"
# CUDA_VISIBLE_DEVICES=7 python main.py --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --num_seeds 1 --dataset_name ccdv/cnn_dailymail --counterfactual_removal True
echo "end 100-shot prompt-tune_summary"

# echo "CONTROLLING EXPS FOR CNNDM - WITH COUNTERFACTUAL TRAINING"
CUDA_VISIBLE_DEVICES=4 python controllability.py --big_testset --dataset_name=xsum --few_shot 100 --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --dataset_name ccdv/cnn_dailymail --counterfactual_trained
