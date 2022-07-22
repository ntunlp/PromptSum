### dataset
dataset="xsum"
k_shot=10


### backbone model
### T5-large backbone
#pretrain_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_full_model_114k"
#pretrain_prompt_ckpt="/data/qin/PromptSumm/Summarization/t5_tagger_pretrained_ckpt_0520bak/bestckpt_prompt_114k"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_c_210k/bestckpt_prompt"
#pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
#pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"
### PEGASUS backbone
pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_340k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/014_c_340k/bestckpt_prompt"


### k-shot
echo "start " + str(k_shot) + "-shot prompt-tune_entity"
python main.py --dataset_name dataset --few_shot k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_entity 60
echo "end " + str(k_shot) + "-shot prompt-tune_entity"

echo "start " + str(k_shot) + "-shot prompt-tune_summary"
python main.py --dataset_name dataset --few_shot k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
echo "end " + str(k_shot) + "-shot prompt-tune_summary"

#echo "start " + str(k_shot) + "-shot full test set eval - uncomment following line and comment out previous two blocks"
#python main.py --dataset_name dataset --few_shot k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --full_testset --max_epoch_summary 0
#echo "end " + str(k_shot) + "-shot full test set eval - uncomment following line and comment out previous two blocks"