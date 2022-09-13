### dataset
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
device="5"
cache='/home/mathieu/hf_models/pegasus-large/'

### backbone model
### T5-large backbone
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone
pretrain_ckpt="/home/mathieu/PromptSumm/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/home/mathieu/PromptSumm/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"



# train + val
#echo "start full-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache
echo "start full-shot prompt-tune_summary"
CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache --reuse_entity_file

# test
#echo "start full-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --cache_path $cache
#echo "start full-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache
