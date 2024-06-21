### dataset
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="10" # in ["1", "10", "100"]
device="0"
n_epochs="2" # 60


############################ PromptSum

##### train + val
echo "start k-shot prompt-tune_entity"
CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --few_shot $k_shot --finetune_entity --max_epoch_entity $n_epochs --prompt_number 100
echo "start k-shot prompt-tune_summary"
CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --few_shot $k_shot --finetune_summary --max_epoch_summary $n_epochs --eval_epoch_0 --prompt_number 100
#### test
echo "start k-shot prompt-tune_entity - TEST SET"
CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_entity --max_epoch_entity 0 --max_epoch_summary 0 --prompt_number 100
echo "start k-shot prompt-tune_summary - TEST SET"
CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --max_epoch_summary 0 --prompt_number 100

############################ PromptSum - no pre-training

#### train + val
#echo "start k-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --few_shot $k_shot --finetune_entity --use_pretrain_ckpt --max_epoch_entity $n_epochs --prompt_number 100
#echo "start k-shot prompt-tune_summary"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --max_epoch_summary $n_epochs --eval_epoch_0 --prompt_number 100
#### test
#echo "start k-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_entity --use_pretrain_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --prompt_number 100
#echo "start k-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --max_epoch_summary 0 --prompt_number 100

############################ PromptSum - no fine-tuned S-prompt

#### test
#echo "PromptSum no fine-tuned S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_finetuned_sprompt

############################ PromptSum - no S-prompt

#### test
#echo "PromptSum no S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_sprompt

############################ PromptSum - no fine-tuned E-prompt

#### test
#echo "PromptSum no E-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_finetuned_eprompt

############################ PromptSum - no entity chain

#### test
#echo "Prompt no entity chain - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --max_epoch_summary 0 --prompt_number 100 --no_entity_chain

############################ PromptSum - oracle

#### train & val
#echo "start k-shot prompt-tune_summary ORACLE"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --few_shot $k_shot --finetune_summary --use_tagger --guidance_mode target --max_epoch_summary $n_epochs --prompt_number 100
#### test
#echo "start k-shot prompt-tune_summary ORACLE - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_tagger --guidance_mode target --max_epoch_summary 0 --prompt_number 100
