# PromptSum
Cool prompting for parameter-efficient few-shot controllable summarization!

# Bash scripts

!!! Don't forget to change the "root" variable at the top of args in main.py!!!

Run the corresponding bash script for each use case:

### Pre-training 
bash run_pretraining.sh

### 0-shot summarization (3 seeds in validation, 1 seed in test)
For <ins>PromptSum</ins>:

bash runall_zeroshot_promptsum.sh

### Few-shot summarization (3 seeds)
For the <ins>baselines</ins>:

bash runall_kshot_baselines.sh

For the <ins>oracle</ins>:

bash runall_kshot_oracle.sh

For <ins>PromptSum</ins>:

bash runall_kshot_promptsum.sh

For <ins>controllability</ins> experiments:

bash runall_kshot_controllability.sh

For <ins>counterfactual</ins> training experiments:

bash runall_kshot_counterfactual.sh

For <ins>hallucinations</ins> experiments:

bash runall_kshot_hallucination.sh

### Full-shot summarization (1 seed)
For <ins>PromptSum</ins>:

bash run_fullshot_promptsum.sh

## Citation

If you find any of this useful, please kindly consider citing our paper in your publication.

```
@article{ravaut2023promptsum,
  title={Promptsum: Parameter-efficient controllable abstractive summarization},
  author={Ravaut, Mathieu and Chen, Hailin and Zhao, Ruochen and Qin, Chengwei and Joty, Shafiq and Chen, Nancy},
  journal={arXiv preprint arXiv:2308.03117},
  year={2023}
}
```
