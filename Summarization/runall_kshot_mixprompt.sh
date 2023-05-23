### dataset
dataset="billsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="100" # in ["1", "10", "100"]
device="3"
cache='/data/mathieu/hf_models/pegasus-large/'

### backbone model
##### T5-large backbone
#pretrain_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_full_model_114k"
#pretrain_prompt_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_prompt_114k"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
# pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
# pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
##### PEGASUS backbone (015_n_400k / 016 / 019)
pretrain_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/019/bestckpt_full_model"
pretrain_prompt_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/019/bestckpt_prompt"


############################ MixPrompt (PromptSum) - no pre-training

#echo "k_shot 1"
##### train + val
#echo "start k-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --finetune_entity --use_pretrain_ckpt --max_epoch_entity 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100
#echo "start k-shot prompt-tune_summary"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --max_epoch_summary 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0 --prompt_number 100
##### test
#echo "start k-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_entity --use_pretrain_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100
#echo "start k-shot prompt-tune_summary - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_pretrain_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100

############################ MixPrompt (PromptSum)

#echo "k_shot 1"
##### train + val
#echo "start k-shot prompt-tune_entity"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100
#echo "start k-shot prompt-tune_summary"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --eval_epoch_0 --prompt_number 100
##### test
#echo "start k-shot prompt-tune_entity - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 0 --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache 
echo "start k-shot prompt-tune_summary - TEST SET"
CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100

############################ MixPrompt (PromptSum) - no fine-tuned S-prompt

##### test
#echo "PromptSum no fine-tuned S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_finetuned_sprompt

############################ MixPrompt (PromptSum) - no S-prompt

##### test
#echo "PromptSum no S-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_sprompt

############################ MixPrompt (PromptSum) - no fine-tuned E-prompt

##### test
#echo "PromptSum no E-prompt - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_finetuned_eprompt

############################ MixPrompt (PromptSum) - no entity chain

##### test
#echo "Prompt no entity chain - Summary inference - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --cache_path $cache --prompt_number 100 --no_entity_chain

############################ MixPrompt - oracle

##### train & val
#echo "start k-shot prompt-tune_summary ORACLE"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --few_shot $k_shot --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100
##### test
#echo "start k-shot prompt-tune_summary ORACLE - TEST SET"
#CUDA_VISIBLE_DEVICES=$device python main.py --model PegasusMixPrompt --dataset_name $dataset --full_testset --few_shot $k_shot --finetune_summary --use_t5_tagger --guidance_mode target --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 0 --model_name google/pegasus-large --use_lm_adapted 0 --cache_path $cache --prompt_number 100

