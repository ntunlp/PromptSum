### dataset
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="100" # in ["1", "10", "100", "full"]
device="1"
ckpt_name="bestckpt_from_pretrained" # ['bestckpt_from_pretrained', 'bestckpt', 'bestckpt_full_weights_from_pretrained']
tagger_ckpt_name="bestckpt_prompt_from_pretrained" # ["bestckpt_prompt", "bestckpt_prompt_from_pretrained"]
# 
declare -A seed_shot_map
seed_shot_map=(100 0 full 42)
seed=$seed_shot_map[$k_shot]
### backbone model
### T5-large backbone
cache='/export/home/cache'
pretrain_ckpt="/export/home/PromptSumm/Summarization/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/export/home/PromptSumm/Summarization/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"
# --big_testset
if [ "$dataset" = "billsum" ];
then
    length=512
    batch_size=16
else
    length=512
    batch_size=16
fi


CUDA_VISIBLE_DEVICES=$device python hallucination.py --max_length $length --valid_size_per_gpu_summary $batch_size --big_testset --seed $seed --ckpt_name $ckpt_name --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt  --cache_path $cache 
