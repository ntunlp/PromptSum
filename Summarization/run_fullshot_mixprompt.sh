### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
device="0"
cache='/home/mathieu/hf_models/pegasus-large/'

### backbone model
##### T5-large backbone
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone
pretrain_ckpt="/home/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/home/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"


############################ MixPrompt (PromptSum) - no pre-training

##### train + val
#echo "start full-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_entity --use_pretrain_ckpt --cache_path $cache
#echo "start full-shot prompt-tune_summary"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --cache_path $cache
##### test
#echo "start full-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --use_pretrain_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --cache_path $cache
#echo "start full-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --max_epoch_summary 0 --cache_path $cache

############################ MixPrompt (PromptSum)

##### train + val
#echo "start full-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache
#echo "start full-shot prompt-tune_summary"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache
##### test
#echo "start full-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --cache_path $cache
#echo "start full-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --reuse_entity_file --

############################ MixPrompt - oracle

##### train & val
#echo "start full-shot prompt-tune_summary ORACLE"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache
##### test
echo "start full-shot prompt-tune_summary ORACLE - TEST SET"
CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --test_on_val

############################ MixPrompt (PromptSum) - no pre-training TUNE WEIGHTS

##### train + val
#echo "start full-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_entity --use_pretrain_ckpt --cache_path $cache --tune_weights
#echo "start full-shot prompt-tune_summary"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --cache_path $cache --tune_weights
##### test
#echo "start full-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --use_pretrain_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --cache_path $cache --tune_weights
#echo "start full-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --max_epoch_summary 0 --cache_path $cache --tune_weights

############################ MixPrompt (PromptSum) TUNE WEIGHTS

##### train + val
#echo "start full-shot prompt-tune_entity TUNE WEIGHTS"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0 --tune_weights
#echo "start full-shot prompt-tune_summary TUNE WEIGHTS"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0 --tune_weights
#### test
#echo "start full-shot prompt-tune_entity TUNE WEIGHTS - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights
#echo "start full-shot prompt-tune_summary TUNE WEIGHTS - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights

############################ MixPrompt - oracle TUNE WEIGHTS

##### train & val
#echo "start full-shot prompt-tune_summary ORACLE TUNE WEIGHTS"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights
##### test
#echo "start full-shot prompt-tune_summary ORACLE TUNE WEIGHTS - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights
