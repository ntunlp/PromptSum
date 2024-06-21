### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="10"
device="0"
bs=4


############################ PromptSum

##### test
echo "start 0-shot prompt-tune_entity - TEST SET"
CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_entity --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0
echo "start 0-shot prompt-tune_summary - TEST SET"
CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --model PegasusMixPrompt --max_test_size 100000 --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_summary --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0

############################ PromptSum - oracle

##### test
#echo "start 0-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_summary --use_t5_tagger --guidance_mode target --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0

