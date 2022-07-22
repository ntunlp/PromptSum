### dataset
dataset="xsum"
k_shot="10"


### backbone model
### T5-large backbone
#pretrain_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_full_model_114k"
#pretrain_prompt_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_prompt_114k"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
### PEGASUS backbone
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_340k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_340k/bestckpt_prompt"


### k-shot
echo "start k-shot baseline-1: all-params finetune summary"
### train & val
python main.py --dataset_name $dataset --few_shot $k_shot --model T5Finetune --finetune_summary --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 60
### test set inference (uncomment and comment out previous line)
#python main.py --dataset_name $dataset --few_shot $k_shot --model T5Finetune --finetune_summary --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --full_testset --max_epoch_summary 0
echo "end k-shot baseline-1: all-params finetune summary"

echo "start k-shot baseline-2: simple prompt-tune summary"
### train & val
python main.py --dataset_name $dataset --few_shot $k_shot --model T5SoftPrompt --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 60
### test set inference (uncomment and comment out previous line)
#python main.py --dataset_name $dataset --few_shot $k_shot --model T5SoftPrompt --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --full_testset --max_epoch_summary 0
echo "end k-shot baseline-2: simple prompt-tune summary"

echo "start k-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
### train & val
python main.py --dataset_name $dataset --few_shot $k_shot --model T5SoftPrompt --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 60
### test set inference (uncomment and comment out previous line)
#python main.py --dataset_name $dataset --few_shot $k_shot --model T5SoftPrompt --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --full_testset --max_epoch_summary 0
echo "end k-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
