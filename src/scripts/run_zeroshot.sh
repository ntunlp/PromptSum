### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="10"
device="3"
bs=4
cache='../../hf_models/pegasus-large/'

### backbone model
##### PEGASUS backbone
pretrain_ckpt="../pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="../pretrained_ckpt/015_n_400k/bestckpt_prompt"

############################ MixPrompt (PromptSum)

##### test
#echo "start 0-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache
echo "start 0-shot prompt-tune_summary - TEST SET"
CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --max_test_size 100000 --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache

############################ MixPrompt - oracle

##### test
#echo "start 0-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache

