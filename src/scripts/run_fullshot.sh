### dataiet
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
device="0"


############################ PromptSum

##### train + val
echo "PromptSum - Training E-prompt"
CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --finetune_entity --prompt_number 100
echo "PromptSum - Training S-prompt"
CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --finetune_summary --prompt_number 100
##### test
#echo "PromptSum - Entity inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_entity --max_epoch_entity 0 --max_epoch_summary 0 --prompt_number 100
#echo "PromptSum - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --max_epoch_summary 0 --prompt_number 100

############################ PromptSum - no pre-training

##### train + val
#echo "PromptSum no pre-training - Training E-prompt"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --finetune_entity --use_pretrain_ckpt --prompt_number 100
#echo "PromptSum no pre-training - Training S-prompt"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --prompt_number 100
##### test
#echo "PromptSum no pre-training - Entity inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_entity --use_pretrain_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --prompt_number 100
#echo "PromptSum no pre-training - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --max_epoch_summary 0 --prompt_number 100

############################ PromptSum - no fine-tuned S-prompt

##### test
#echo "PromptSum no fine-tuned S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_finetuned_sprompt

############################ PromptSum - no S-prompt

##### test
#echo "PromptSum no S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_sprompt

############################ PromptSum - no fine-tuned E-prompt

##### test
#echo "PromptSum no E-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_finetuned_eprompt

############################ PromptSum - no entity chain

##### test
#echo "PromptSum no entity chain - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_entity_chain

############################ PromptSum - oracle

##### train & val
#echo "PromptSum oracle entities - Training S-prompt"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --finetune_summary --use_tagger --guidance_mode target --prompt_number 100
##### test
#echo "PromptSum oracle entities - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --dataset_name $dataset --full_testset --finetune_summary --use_tagger --guidance_mode target --max_epoch_summary 0 --prompt_number 100

