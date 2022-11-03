### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="100" # in ["1", "10", "100"]
device="2"
bs=8
size=50
cache='/data/mathieu/hf_models/pegasus-large/'

### backbone model
# #### PEGASUS backbone
pretrain_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"


############################ Baseline v1: Fine-tuning

##### test
echo "start k-shot baseline-1: all-params finetune summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python human_eval.py --model PegasusFinetune --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --max_test_size $size --valid_size_per_gpu_summary $bs

############################ MixPrompt (PromptSum)

##### test
echo "start k-shot prompt-tune_summary - TEST SET"
CUDA_VISIBLE_DEVICES=$device python human_eval.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --max_test_size $size --valid_size_per_gpu_summary $bs

