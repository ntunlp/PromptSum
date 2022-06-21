#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"


## 10-shot
echo "start 10-shot baseline-1: all-params finetune summary"
#python main.py --few_shot 10 --model T5Finetune --finetune_summary --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 60
#python main.py --few_shot 10 --model T5Finetune --finetune_summary --use_pretrain_ckp --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --full_testset --max_epoch_summary 0
echo "end 10-shot baseline-1: all-params finetune summary"

echo "start 10-shot baseline-2: simple prompt-tune summary"
# CUDA_VISIBLE_DEVICES=7 python main.py --few_shot 10 --model T5SoftPrompt --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --finetune_summary
echo "end 10-shot baseline-2: simple prompt-tune summary"

echo "evaluate 10-shot baseline-2: simple prompt-tune summary"
CUDA_VISIBLE_DEVICES=5 python main.py --few_shot 10 --model T5SoftPrompt --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --finetune_summary --max_epoch_summary 0 --full_testset
echo "end 10-shot baseline-2: simple prompt-tune summary"

echo "start 10-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
#python main.py --few_shot 10 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 10-shot baseline-3: simple prompt-tune summary with pretrained ckpt"

echo "start 10-shot baseline-4: mix prompt-tune summary with no entity chain"
#python main.py --few_shot 10 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 10-shot baseline-4: simple prompt-tune summary with no entity chain"


## 64-shot
echo "start 64-shot baseline-1: all-params finetune summary"
#python main.py --few_shot 64 --model T5Finetune --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 64-shot baseline-1: all-params finetune summary"

echo "start 64-shot baseline-2: simple prompt-tune summary"
# CUDA_VISIBLE_DEVICES=7 python main.py --few_shot 64 --model T5SoftPrompt --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --finetune_summary
echo "end 64-shot baseline-2: simple prompt-tune summary"

echo "start 64-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
#python main.py --few_shot 64 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 64-shot baseline-3: simple prompt-tune summary with pretrained ckpt"

echo "start 64-shot baseline-4: mix prompt-tune summary with no entity chain"
#python main.py --few_shot 64 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 64-shot baseline-4: simple prompt-tune summary with no entity chain"


## 100-shot
echo "start 100-shot baseline-1: all-params finetune summary"
#python main.py --few_shot 100 --model T5Finetune --finetune_summary --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --max_epoch_summary 60
python main.py --few_shot 100 --model T5Finetune --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy --full_testset --max_epoch_summary 0
echo "end 100-shot baseline-1: all-params finetune summary"

echo "start 100-shot baseline-2: simple prompt-tune summary"
#python main.py --few_shot 100 --model T5SoftPrompt --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 100-shot baseline-2: simple prompt-tune summary"

echo "start 100-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
#python main.py --few_shot 100 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 100-shot baseline-3: simple prompt-tune summary with pretrained ckpt"

echo "start 100-shot baseline-4: mix prompt-tune summary with no entity chain"
#python main.py --few_shot 100 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 100-shot baseline-4: simple prompt-tune summary with no entity chain"
