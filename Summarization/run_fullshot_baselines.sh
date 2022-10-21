### dataset
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
device="3"
cache='/data/mathieu/hf_models/pegasus-large/'

### backbone model
##### T5-large backbone
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone
pretrain_ckpt="/data/mathieu/PromptSumm/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/mathieu/PromptSumm/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"


########################### Baseline v1: Fine-tuning

##### train & val
#echo "start full-shot baseline-1: all-params finetune summary"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusFinetune --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --label_smoothing 0.1
##### test
#echo "start full-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusFinetune --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache

############################ Baseline v2: Soft prompt tuning

##### train & val
#echo "start full-shot baseline-2: simple prompt-tune summary"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0
##### test
#echo "start full-shot baseline-2: simple prompt-tune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache

############################ Baseline v3: Soft prompt tuning from our pre-trained checkpoint

##### train & val
#echo "start full-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0
##### test
#echo "start full-shot baseline-3: simple prompt-tune summary with pretrained ckpt - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache

############################ Baseline v4: Soft prompt tuning with TUNE WEIGHTS

# ##### train & val
#echo "start full-shot baseline-4: simple prompt-tune summary TUNE WEIGHTS"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights --valid_size_per_gpu_summary 2 --batch_size_per_gpu_summary 1 --gradient_accumulation_steps_summary 256 
# ##### test
echo "start full-shot baseline-4: simple prompt-tune summary TUNE WEIGHTS - TEST SET"
CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights

############################ Baseline v5: Soft prompt tuning from our pre-trained checkpoint TUNE WEIGHTS

##### train & val
#echo "start full-shot baseline-5: simple prompt-tune summary with pretrained ckpt TUNE WEIGHTS"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0 --tune_weights
##### test
#echo "start full-shot baseline-5: simple prompt-tune summary with pretrained ckpt TUNE WEIGHTS - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_full_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --tune_weights 

