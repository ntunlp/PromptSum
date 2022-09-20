### dataset
dataset="ccdv/cnn_dailymail"
k_shot="100"
device="2"

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
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_prompt"
cache='/data/ruochen/hf_models/pegasus-large/'


# # k-shot
# echo "start k-shot prompt-tune_entity"
# python main.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# echo "end k-shot prompt-tune_entity"

# echo "start k-shot prompt-tune_summary"
# python main.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
# echo "end k-shot prompt-tune_summary"

# echo "start CONTROLLING experiments"
# # train & val
# python controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# # test
# #python controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --big_testset
# echo "end CONTROLLING experiments"


echo "start ENTITY SUCCESS experiments"
# train & val
CUDA_VISIBLE_DEVICES=$device python entity_success.py --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache
# test
echo "end CONTROLLING experiments"

