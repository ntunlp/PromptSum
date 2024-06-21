### dataset
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="100" # in ["1", "10", "100", "full"]
device="7"
ckpt_name="bestckpt_100_from_pretrained" # ['bestckpt_from_pretrained', 'bestckpt', 'bestckpt_full_weights_from_pretrained']
# 
declare -A dataset_name_map
dataset_name_map=(ccdv/cnn_dailymail cnndm xsum xsum billsum billsum samsum samsum)
declare -A seed_shot_map
seed_shot_map=(100 0 full 42)
seed=$seed_shot_map[$k_shot]
### backbone model
### T5-large backbone
cache='../../hf_models/'
pretrain_ckpt="../pretrained_ckpt/019/bestckpt_full_model"
pretrain_prompt_ckpt="../pretrained_ckpt/019/bestckpt_prompt"
# --big_testset
if [ "$dataset" = "billsum" ];
then
    length=768
    batch_size=16
else
    length=512
    batch_size=16
fi

dataset_name=$dataset_name_map[$dataset]
log_file=log_acl_controlling/${dataset_name}/hallucination\_$k_shot\_promptsumm_1.log

echo $log_file
CUDA_VISIBLE_DEVICES=$device python hallucination.py --max_length $length --valid_size_per_gpu_summary $batch_size --seed $seed --ckpt_name $ckpt_name --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache \
2>&1 | tee -a $log_file
