cache='/export/home/cache'
pretrain_ckpt="../pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="../pretrained_ckpt/015_n_400k/bestckpt_prompt"

### parameters to change
dataset="xsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
dataset_name="xsum" # in ["cnndm", "xsum", "billsum", "samsum"]
seed="0" # in [0, 0, 1, 1]
device="7" 
model="PegasusMixPrompt" # ["PegasusMixPrompt", "CTRLsum", "CTRLsum_origin"]
k_shot="100"
batch_size="16"
ckpt_name="bestckpt_from_pretrained" # ['bestckpt_from_pretrained', 'bestckpt', 'bestckpt_full_weights_from_pretrained']

mode="interactive"

CUDA_VISIBLE_DEVICES=$device python src/entity_success.py --mode $mode --ckpt_name $ckpt_name --seed $seed --valid_size_per_gpu_summary $batch_size --model $model --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache

