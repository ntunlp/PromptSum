### dataset
dataset="xsum"
k_shot="10"

echo "start k-shot prompt-tune_summary for cnndm with counterfactual training"
python src/main_few_shot.py --dataset $dataset --num_seeds 1 --few_shot $k_shot --finetune_summary --max_epoch_summary 60 --counterfactual_removal True
echo "end k-shot prompt-tune_summary for cnndm with counterfactual training"

echo "start CONTROLLING experiments with counterfactual training"
python src/controllability.py --dataset $dataset --few_shot $k_shot --counterfactual_trained --big_testset
echo "end CONTROLLING experiments with counterfactual training"