### dataset
dataset="samsum"
k_shot="10"
device="5"

### backbone model
### T5-large backbone
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
### PEGASUS backbone
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_prompt"

### k-shot
echo "start 0-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --num_seeds 3 --zero_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --model_name google/pegasus-large --use_lm_adapted 0 --cache_path /data/mathieu/hf_models/pegasus-large/
echo "end 0-shot prompt-tune_entity"

echo "start 0-shot prompt-tune_summary"
CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --num_seeds 3 --zero_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --model_name google/pegasus-large --use_lm_adapted 0 --cache_path /data/mathieu/hf_models/pegasus-large/
echo "end 0-shot prompt-tune_summary"
