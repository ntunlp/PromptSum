### dataset
dataset="xsum"
device="3"

### backbone model
### T5-large backbone
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_prompt"

# entity
echo "start full-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path /data/mathieu/hf_models/pegasus-large/
echo "end full-shot prompt-tune_entity"

# summary
echo "start full-shot prompt-tune_summary"
CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --reuse_entity_file --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path /data/mathieu/hf_models/pegasus-large/
echo "end full-shot prompt-tune_summary"
