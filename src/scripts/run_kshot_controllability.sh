### parameters to change
dataset="samsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
device="7"
model="PegasusMixPrompt" # ["PegasusMixPrompt", "CTRLsum", "CTRLsum_origin"]
ckpt_name="bestckpt_100_from_pretrained" # ['bestckpt_from_pretrained', 'bestckpt', 'bestckpt_full_weights_from_pretrained']

declare -A dataset_name_map
dataset_name_map=(ccdv/cnn_dailymail cnndm xsum xsum billsum billsum samsum samsum)

# k_shot="100" # in ["1", "10", "100", "full"]
# mode="k_entity_test" #["oracle", "oracle_add_entity", "oracle_drop_entity", "single_entity_test", "k_entity_test", "interactive"]



declare -A valid_batch_size_map
valid_batch_size_map=(PegasusMixPrompt 16 CTRLsum_origin 64)
batch_size=$valid_batch_size_map[$model]
declare -A seed_shot_map
# seed_shot_map=(100 0 full 42)
seed_shot_map=(cnndm 0 xsum 0 billsum 1 samsum 1)

# k_shot="100" # in ["1", "10", "100", "full"]
# mode="k_entity_test" #["oracle", "oracle_add_entity", "oracle_drop_entity", "single_entity_test", "k_entity_test", "interact
# # k-shot
# echo "start k-shot prompt-tune_entity"
# python src/main_few_shot.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_entity --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# echo "end k-shot prompt-tune_entity"

# echo "start k-shot prompt-tune_summary"
# python src/main_few_shot.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_summary --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --max_epoch_summary 60
# echo "end k-shot prompt-tune_summary"

# echo "start CONTROLLING experiments"
# # train & val
# python src/controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt
# # test
# #python src/controllability.py --dataset $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt  --pretrain_prompt_ckpt $pretrain_prompt_ckpt --big_testset
# echo "end CONTROLLING experiments"
pretrain_suffix=''
if [ $ckpt_name != "bestckpt_from_pretrained" ]
then
    pretrain_suffix='_no_pretrain'
fi

for k_shot in "100" #"100" "full"
do
    # for dataset in "ccdv/cnn_dailymail" "xsum" "billsum" "samsum"
    # do
        dataset_name=$dataset_name_map[$dataset]
        seed=$seed_shot_map[$dataset_name]
        CTRLsum_ckpt_dir="/export/home/ctrl-sum/${dataset_name}_ctrlsum_100" # ['cnndm_ctrlsum_100', 'xsum_ctrlsum_100', 'cnndm_ctrlsum']
        echo "CTRL ckpt: ", $CTRLsum_ckpt_dir, "seed: " $seed
        for k_entity in 0 1 2 5
        do
            if [ $k_entity -eq 0 ]
            then
                mode="oracle"
            else
                mode="k_entity_test"
            fi
            log_file=log_acl_controlling/${dataset_name}/$model\_$k_shot$pretrain_suffix\_$mode\_k=$k_entity.log
            echo "\n\nstart ENTITY SUCCESS experiments: K=", $k_entity, " log: " $log_file
            # train & val
            CUDA_VISIBLE_DEVICES=$device python entity_success.py --mode $mode --k_entity $k_entity --ckpt_name $ckpt_name --seed $seed --valid_size_per_gpu_summary $batch_size --model $model --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --CTRLsum_ckpt_dir $CTRLsum_ckpt_dir --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache 2>&1 \
            | tee -a $log_file
            # test
            echo "end CONTROLLING experiments"
        done
    # done
done