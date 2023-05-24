### dataset
dataset="xsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="100" # 100
device=2
bs=8
cache='../../hf_models/pegasus-large/'

### backbone model
##### PEGASUS backbone
pretrain_ckpt="../pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="../015_n_400k/bestckpt_prompt"


######################################################## 100-shot

############################ Baseline v1: Fine-tuning

##### test
echo "**************************************************************"
#echo "start k-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --model PegasusFinetune --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs --seeds_to_keep 1,2
############################ Baseline v3: Soft prompt tuning with pre-training

##### test
echo "**************************************************************"
#echo "start k-shot baseline-3: simple prompt-tune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs --seeds_to_keep 1,2

############################ MixPrompt

##### test
echo "**************************************************************"
#echo "start k-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs
echo "start k-shot prompt-tune_summary - TEST SET"
CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs --seeds_to_keep 1,2

######################################################## Full-shot

############################ Baseline v1: Fine-tuning

##### test
echo "**************************************************************"
#echo "start full-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusFinetune --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs

############################ Baseline v2: Soft prompt tuning

##### test
echo "**************************************************************"
#echo "start full-shot baseline-2: simple prompt-tune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs

############################ MixPrompt

##### test
echo "**************************************************************"
#echo "start full-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --cache_path $cache --valid_size_per_gpu_summary $bs
#echo "start full-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --valid_size_per_gpu_summary $bs
