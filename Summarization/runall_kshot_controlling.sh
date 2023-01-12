cache='/data/mathieu/hf_models/pegasus-large/'
pretrain_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_full_model"
pretrain_prompt_ckpt="/data/mathieu/PromptSum/t5_tagger_pretrained_ckpt/015_n_400k/bestckpt_prompt"

### parameters to change
dataset="xsum" # in ["ccdv/cnn_dailymail", "xsum", "billsum", "samsum"]
dataset_name="xsum" # in ["cnndm", "xsum", "billsum"]
seed="0" # in [0, 0, 1, 1]
device="1" 
model="PegasusMixPrompt" # ["PegasusMixPrompt", "CTRLsum", "CTRLsum_origin"]
batch_size="16"
ckpt_name="bestckpt_from_pretrained" # ['bestckpt_from_pretrained', 'bestckpt', 'bestckpt_full_weights_from_pretrained']

# k_shot="100" # in ["1", "10", "100", "full"]
# mode="k_entity_test" #["oracle", "oracle_add_entity", "oracle_drop_entity", "single_entity_test", "k_entity_test", "interact
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
pretrain_suffix=''
if [ $ckpt_name != "bestckpt_from_pretrained" ]
then
    pretrain_suffix='_no_pretrain'
fi 

for k_shot in "100" #"100" "full"
do
    # for dataset in "ccdv/cnn_dailymail" "xsum" "billsum" "samsum"
    # do 
        #dataset_name=$dataset_name_map[$dataset]
        #seed=$seed_shot_map[$dataset_name]
        #CTRLsum_ckpt_dir="/export/home/ctrl-sum/${dataset_name}_ctrlsum_100" # ['cnndm_ctrlsum_100', 'xsum_ctrlsum_100', 'cnndm_ctrlsum']
        #echo "CTRL ckpt: ", $CTRLsum_ckpt_dir, "seed: " $seed
        echo "dataset: " $dataset
	echo "dataset name: " $dataset_name
	echo "seed: " $seed
	mode="interactive"
	for k_entity in 3
        do
            log_file="log/temp.log"
            echo "start ENTITY SUCCESS experiments " $log_file
            # train & val
            CUDA_VISIBLE_DEVICES=$device python entity_success.py --mode $mode --k_entity $k_entity --ckpt_name $ckpt_name --seed $seed --valid_size_per_gpu_summary $batch_size --model $model --dataset_name $dataset --few_shot $k_shot --pretrain_ckpt $pretrain_ckpt --pretrain_prompt_ckpt $pretrain_prompt_ckpt --cache_path $cache 2>&1 \
            | tee -a $log_file
            # test
            echo "end CONTROLLING experiments"
        done
    # done
done
