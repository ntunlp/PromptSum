### dataiet
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
device="7"
cache='../../hf_models/pegasus-large/'


### backbone model
##### PEGASUS backbone (015_n_400k / 016)
pretrain_ckpt="../pretrained_ckpt/019/bestckpt_full_model"
pretrain_prompt_ckpt="../pretrained_ckpt/019/bestckpt_prompt"


############################ MixPrompt (PromptSum)

##### train + val
#echo "PromptSum - Training E-prompt"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache --prompt_number 100
#echo "PromptSum - Training S-prompt"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache --prompt_number 100
##### test
#echo "PromptSum - Entity inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --cache_path $cache --prompt_number 100
echo "PromptSum - Summary inference - TEST SET"
CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100

############################ MixPrompt (PromptSum) - no pre-training

##### train + val
#echo "PromptSum no pre-training - Training E-prompt"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --finetune_entity --use_pretrain_ckpt --cache_path $cache --prompt_number 100
#echo "PromptSum no pre-training - Training S-prompt"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --cache_path $cache --prompt_number 100
##### test
#echo "PromptSum no pre-training - Entity inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_entity --use_pretrain_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --cache_path $cache --prompt_number 100
#echo "PromptSum no pre-training - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100

############################ MixPrompt (PromptSum) - no fine-tuned S-prompt

##### test
#echo "PromptSum no fine-tuned S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_finetuned_sprompt

############################ MixPrompt (PromptSum) - no S-prompt

##### test
#echo "PromptSum no S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_sprompt

############################ MixPrompt (PromptSum) - no fine-tuned E-prompt

##### test
#echo "PromptSum no E-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_finetuned_eprompt

############################ MixPrompt (PromptSum) - no entity chain

##### test
#echo "PromptSum no entity chain - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_entity_chain

############################ MixPrompt - oracle

##### train & val
#echo "PromptSum oracle entities - Training S-prompt"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --finetune_summary --use_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache --prompt_number 100
##### test
#echo "PromptSum oracle entities - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --use_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100

