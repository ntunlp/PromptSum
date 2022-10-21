cache='/export/home/cache'
pretrain_ckpt="/export/home/PromptSumm/Summarization/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/export/home/PromptSumm/Summarization/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"

### parameters to change
dataset="ccdv/cnn_dailymail" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
k_shot="100" # in ["1", "10", "100", "full"]
device="0"
model="PegasusMixPrompt" # ["PegasusMixPrompt", "CTRLsum", "CTRLsum_origin"]
ckpt_name="bestckpt_from_pretrained" # ['bestckpt_from_pretrained', 'bestckpt', 'bestckpt_full_weights_from_pretrained']
CTRLsum_ckpt_dir='/export/home/ctrl-sum/cnndm_ctrlsum_100' # ['cnndm_ctrlsum_100', 'xsum_ctrlsum_100', 'cnndm_ctrlsum']
mode="interactive" #["oracle", "oracle_add_entity", "oracle_drop_entity", "single_entity_test", "k_entity_test", "interactive"]
k_entity=2

declare -A valid_batch_size_map
valid_batch_size_map=(PegasusMixPrompt 16 CTRLsum_origin 64)
batch_size=$valid_batch_size_map[$model]
declare -A seed_shot_map
seed_shot_map=(100 0 full 42)
seed=$seed_shot_map[$k_shot]
# # k-shot
# echo "start k-shot prompt-tune_entity"
# python main.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# echo "end k-shot prompt-tune_entity"

# echo "start k-shot prompt-tune_summary"
# python main.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
# echo "end k-shot prompt-tune_summary"

# echo "start CONTROLLING experiments"
# # train & val
# python controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# # test
# #python controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --big_testset
# echo "end CONTROLLING experiments"


echo "start ENTITY SUCCESS experiments"
# train & val
CUDA_VISIBLE_DEVICES=$device python entity_success.py --mode $mode --k_entity $k_entity --ckpt_name $ckpt_name --seed $seed --valid_size_per_gpu_summary $batch_size --model $model --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache
# test
echo "end CONTROLLING experiments"

