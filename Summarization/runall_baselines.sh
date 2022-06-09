#echo "start 64-shot baseline-1: all-params finetune summary"
#python main.py --few_shot 64 --model T5Finetune --lr_summary 5e-5 --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
#echo "end 64-shot baseline-1: all-params finetune summary"

#echo "start 64-shot baseline-2: simple prompt-tune summary"
#python main.py --few_shot 64 --model T5SoftPrompt --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
#echo "end 64-shot baseline-2: simple prompt-tune summary"

echo "start 64-shot baseline-3: simple prompt-tune summary with pretrained ckpt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
python main.py --few_shot 64 --model T5SoftPrompt --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 64-shot baseline-3: simple prompt-tune summary with pretrained ckpt"

#echo "start 10-shot baseline-4 (SAME AS baseline-v3): mix prompt-tune summary with no entity chain"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
#python main.py --few_shot 10 --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
#echo "end 10-shot baseline-4: mix prompt-tune summary with no entity chain"
