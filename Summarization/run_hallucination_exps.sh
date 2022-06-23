pretrain_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/hailin/PromptSumm/t5_tagger_pretrained_ckpt/012_cc_ent_v2_120k/012_cc_ent_v2_120k/bestckpt_prompt"

#  --big_testset
CUDA_VISIBLE_DEVICES=4 python hallucination.py --dataset_name=xsum  --big_testset --few_shot 100 --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt
