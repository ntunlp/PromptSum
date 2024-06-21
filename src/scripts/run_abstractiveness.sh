### dataset
dataset="xsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="10" # 100
device=0
bs=8


######################################################## 100-shot

############################ Fine-tuning

##### test
#echo "start k-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --model PegasusFinetune --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --valid_size_per_gpu_summary $bs --seeds_to_keep 1,2

############################ PromptSum

##### test
#echo "start k-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_entity --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --valid_size_per_gpu_summary $bs
#echo "start k-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --valid_size_per_gpu_summary $bs --seeds_to_keep 1,2

######################################################## Full-shot

############################ Fine-tuning

##### test
#echo "start full-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --model PegasusFinetune --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --valid_size_per_gpu_summary $bs

############################ PromptSum

##### test
#echo "start full-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --max_epoch_entity 0 --max_epoch_summary 0 --valid_size_per_gpu_summary $bs
#echo "start full-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python src/main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --max_epoch_summary 0 --valid_size_per_gpu_summary $bs
