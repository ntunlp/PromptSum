#pretrain_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_full_model_114k"
#pretrain_prompt_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_prompt_114k"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"


## 10-shot
echo "start 10-shot prompt-tune_entity"
#python main.py --few_shot 10 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 60
echo "end 10-shot prompt-tune_entity"

echo "start 10-shot prompt-tune_summary"
#python main.py --few_shot 10 --finetune_summary --infer_val_entities --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
#python main.py --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
python main.py --few_shot 10 --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --full_testset --max_epoch_summary 0 --dataset_name xsum
echo "end 10-shot prompt-tune_summary"

echo "start 10-shot prompt-tune_summary oracle"
#python main.py --few_shot 10 --finetune_summary --guidance_mode target --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
echo "end 10-shot prompt-tune_summary oracle"


## 64-shot
echo "start 64-shot prompt-tune_entity"
#python main.py --few_shot 64 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
echo "end 64-shot prompt-tune_entity"

echo "start 64-shot prompt-tune_summary"
#python main.py --few_shot 64 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt 
echo "end 64-shot prompt-tune_summary"

echo "start 64-shot prompt-tune_summary oracle"
#python main.py --few_shot 64 --finetune_summary --guidance_mode target --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
echo "end 64-shot prompt-tune_summary oracle"


## 100-shot
echo "start 100-shot prompt-tune_entity"
#python main.py --few_shot 100 --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
echo "end 100-shot prompt-tune_entity"

echo "start 100-shot prompt-tune_summary"
#python main.py --few_shot 100 --finetune_summary --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt 
echo "end 100-shot prompt-tune_summary"

echo "start 100-shot prompt-tune_summary oracle"
#python main.py --few_shot 100 --finetune_summary --guidance_mode target --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
echo "end 100-shot prompt-tune_summary oracle"
