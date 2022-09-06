### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "samsum", "billsum"]
device="4"

### backbone model
### T5-large backbone
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_1070k/bestckpt_prompt"
cache='/data/mathieu/hf_models/pegasus-large/'


##### train & val
echo "start full-shot baseline-1: all-params finetune summary"
CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusFinetune --dataset_name $dataset --finetune_summary --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache
#### test
#echo "start full-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusFinetune --dataset_name $dataset --full_testset --finetune_summary --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache


#echo "start full-shot baseline-2: simple prompt-tune summary"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache
# #### test
# echo "start full-shot baseline-2: simple prompt-tune summary - TEST SET"
# CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache


# echo "start full-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
# CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0  --cache_path $cache
# #### test
# echo "start full-shot baseline-3: simple prompt-tune summary with pretrained ckpt - TEST SET"
# CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0  --cache_path $cache
