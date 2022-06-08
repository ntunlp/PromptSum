echo "start 10-shot all-params finetune summary"
python main.py --few_shot 10 --model T5Finetune --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 10-shot all-params finetune summary"

echo "start 10-shot baseline prompt-tune summary"
python main.py --few_shot 10 --model T5SoftPrompt --finetune_summary --use_pretrain_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 10-shot baseline prompt-tune summary"

echo "start 10-shot prompt-tune summary with pretrained ckpt"
pretrain_ckpt = "/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
pretrain_prompt_ckpt = "/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
python main.py --few_shot 10 --model T5SoftPrompt --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --infer_val_entities --use_entity_chain --use_t5_tagger --if_spacy
echo "end 10-shot prompt-tune summary with pretrained ckpt"