### dataset
dataset="xsum"
k_shot="10"
device="5"

### backbone model
### T5-large backbone
#pretrain_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_full_model_114k"
#pretrain_prompt_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_prompt_114k"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
### k-shot
# echo "start k-shot prompt-tune_entity"
# CUDA_VISIBLE_DEVICES=$device python main.py --dataset_name $dataset --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 60
# echo "end k-shot prompt-tune_entity"

# echo "start k-shot prompt-tune_summary"
# CUDA_VISIBLE_DEVICES=$device python main.py --dataset_name $dataset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
# echo "end k-shot prompt-tune_summary"


### PEGASUS backbone
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_340k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_340k/bestckpt_prompt"
### k-shot
echo "start k-shot prompt-tune_entity"
# CUDA_VISIBLE_DEVICES=$device python main.py --dataset_name $dataset --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path /data/ruochen/hf_models/pegasus-large/
echo "end k-shot prompt-tune_entity"

echo "start k-shot prompt-tune_summary"
CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path /data/ruochen/hf_models/pegasus-large/
echo "end k-shot prompt-tune_summary"