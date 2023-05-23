### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="10"
device="3"
bs=4
cache='/data/mathieu/hf_models/pegasus-large/'

### backbone model
##### T5-large backbone
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone
pretrain_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"

############################ Baseline v1: Fine-tuning

###### test
#echo "start k-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --max_test_size 100 --model PegasusFinetune --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --valid_size_per_gpu_summary $bs

########################### Baseline v3: Soft prompt tuning from our pre-trained checkpoint

##### test
#echo "start k-shot baseline-3: simple prompt-tune summary with pretrained ckpt - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main_few_shot.py --model PegasusSoftPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --num_seeds 1 --zero_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache

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

